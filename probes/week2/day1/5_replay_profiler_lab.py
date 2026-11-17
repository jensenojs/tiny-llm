"""Replay-profiler credibility lab.

`benches/profile_week2_kernels.py` never executes the model. It rebuilds one
decode step's kernels on synthetic inputs, wraps each group's `mx.eval` in
`perf_counter`, and reports the four groups' shares. The method trades the real
execution path for decomposability and repeatability. This script bounds when
that trade is sound.

Comparisons, all in one process:

1. build() only vs build() + mx.eval -- what the shares measure when nothing runs
2. fixed order vs rotated order -- who pays the first-run cost
3. mean vs median over the same samples -- how much the statistic matters
4. one real decode step's wall clock vs the sum of the replay medians
5. the real forward with a forced sync after every stage -- an ordering
   reference whose bias is printed alongside it
6. the same harness at E = 64 / 256 / 1024 / 2048 -- where the replay sum and
   the shares stop being dominated by the cost of one mx.eval round trip

Run from the repository root:

    PYTHONPATH=$PWD:$PWD/src .venv/bin/python probes/week2/day1/5_replay_profiler_lab.py
"""

import statistics
from time import perf_counter

import mlx.core as mx


CFG = {
    "layers": 2,
    "hidden": 64,
    "heads": 4,
    "kv_heads": 2,
    "head_dim": 16,
    "intermediate": 128,
    "vocab": 512,
    "context": 8,
    "seed": 20261109,
}
DTYPE = mx.bfloat16
WARMUP = 5
ITERATIONS = 101
GROUPS = ("projections", "attention", "pointwise", "cache")
SWEEP = (64, 256, 1024, 2048)


def make_weight(shape, scale=0.02):
    return (mx.random.normal(shape) * scale).astype(DTYPE)


def rms_norm(x, weight):
    return mx.fast.rms_norm(x, weight, 1e-5)


def silu(x):
    return x * mx.sigmoid(x)


def linear(x, weight):
    return mx.matmul(x, weight.T)


class LayerWeights:
    def __init__(self, cfg):
        e = cfg["hidden"]
        h = cfg["heads"]
        g = cfg["kv_heads"]
        d = cfg["head_dim"]
        i = cfg["intermediate"]
        self.wq = make_weight((h * d, e))
        self.wk = make_weight((g * d, e))
        self.wv = make_weight((g * d, e))
        self.wo = make_weight((e, h * d))
        self.w_gate = make_weight((i, e))
        self.w_up = make_weight((i, e))
        self.w_down = make_weight((e, i))
        self.ln1 = mx.ones((e,), dtype=DTYPE)
        self.ln2 = mx.ones((e,), dtype=DTYPE)
        self.q_norm = mx.ones((d,), dtype=DTYPE)
        self.k_norm = mx.ones((d,), dtype=DTYPE)


class TinyModel:
    """Qwen3-shaped decoder: embedding -> layers(attn + MLP, both residual) -> norm -> lm_head."""

    def __init__(self, cfg):
        mx.random.seed(cfg["seed"])
        self.cfg = cfg
        self.embedding = make_weight((cfg["vocab"], cfg["hidden"]))
        self.layers = [LayerWeights(cfg) for _ in range(cfg["layers"])]
        self.norm = mx.ones((cfg["hidden"],), dtype=DTYPE)
        self.lm_head = make_weight((cfg["vocab"], cfg["hidden"]))
        arrays = [self.embedding, self.norm, self.lm_head]
        for layer in self.layers:
            arrays.extend(layer.__dict__.values())
        mx.eval(*arrays)
        self.parameter_count = sum(a.size for a in arrays)
        self.parameter_bytes = sum(a.size * a.itemsize for a in arrays)

    def forward(self, tokens, cache=None, offset=0, emit=None):
        """Return (logits of the last position, new cache).

        `emit(group, array)` is called after every stage. In timed mode it
        evaluates the array, which forces the queue to drain at that point.
        """
        cfg = self.cfg
        rows = tokens.shape[1]
        d = cfg["head_dim"]
        scale = d**-0.5
        emit = emit or (lambda group, value: None)

        hidden = self.embedding[tokens]
        emit("pointwise", hidden)
        new_cache = []
        for index, layer in enumerate(self.layers):
            normed = rms_norm(hidden, layer.ln1)
            emit("pointwise", normed)

            query = linear(normed, layer.wq)
            key = linear(normed, layer.wk)
            value = linear(normed, layer.wv)
            emit("projections", query)
            emit("projections", key)
            emit("projections", value)

            query = query.reshape(1, rows, cfg["heads"], d).transpose(0, 2, 1, 3)
            key = key.reshape(1, rows, cfg["kv_heads"], d).transpose(0, 2, 1, 3)
            value = value.reshape(1, rows, cfg["kv_heads"], d).transpose(0, 2, 1, 3)

            query = rms_norm(query, layer.q_norm)
            key = rms_norm(key, layer.k_norm)
            query = mx.fast.rope(
                query, d, traditional=False, base=10000.0, scale=1.0, offset=offset
            )
            key = mx.fast.rope(
                key, d, traditional=False, base=10000.0, scale=1.0, offset=offset
            )
            emit("pointwise", query)
            emit("pointwise", key)

            if cache is not None:
                previous_key, previous_value = cache[index]
                key = mx.concatenate([previous_key, key], axis=2)
                value = mx.concatenate([previous_value, value], axis=2)
            emit("cache", key)

            mask = "causal" if rows > 1 else None
            attention = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=scale, mask=mask
            )
            emit("attention", attention)

            attention = attention.transpose(0, 2, 1, 3).reshape(
                1, rows, cfg["heads"] * d
            )
            projected = linear(attention, layer.wo)
            emit("projections", projected)
            hidden = hidden + projected
            emit("pointwise", hidden)

            normed = rms_norm(hidden, layer.ln2)
            emit("pointwise", normed)
            gate = linear(normed, layer.w_gate)
            up = linear(normed, layer.w_up)
            emit("projections", gate)
            emit("projections", up)
            activated = silu(gate) * up
            emit("pointwise", activated)
            down = linear(activated, layer.w_down)
            emit("projections", down)
            hidden = hidden + down
            emit("pointwise", hidden)

            new_cache.append((key, value))

        last = rms_norm(hidden[:, -1:, :], self.norm)
        emit("pointwise", last)
        logits = linear(last, self.lm_head)
        emit("projections", logits)
        return logits, new_cache


class KernelReplay:
    """Synthetic rebuild of one step's groups, mirroring the bench's structure."""

    def __init__(self, model, rows, context):
        cfg = model.cfg
        self.model = model
        self.rows = rows
        self.context = context
        self.phase = "prefill" if rows > 1 else "decode"
        e = cfg["hidden"]
        h = cfg["heads"]
        g = cfg["kv_heads"]
        d = cfg["head_dim"]
        i = cfg["intermediate"]

        mx.random.seed(cfg["seed"] + 1)
        self.hidden = mx.random.normal((1, rows, e)).astype(DTYPE)
        self.query = mx.random.normal((1, h, rows, d)).astype(DTYPE)
        self.key = mx.random.normal((1, g, context, d)).astype(DTYPE)
        self.value = mx.random.normal(self.key.shape).astype(DTYPE)
        self.query_rows = self.query.transpose(0, 2, 1, 3)
        self.key_rows = mx.random.normal((1, rows, g, d)).astype(DTYPE)
        self.gate = mx.random.normal((1, rows, i)).astype(DTYPE)
        self.up = mx.random.normal(self.gate.shape).astype(DTYPE)
        self.tokens = mx.random.randint(0, cfg["vocab"], (1, rows)).astype(mx.int32)
        mx.eval(
            self.hidden,
            self.query,
            self.key,
            self.value,
            self.query_rows,
            self.key_rows,
            self.gate,
            self.up,
            self.tokens,
        )

    def projections(self):
        outputs = []
        hidden = self.hidden
        for layer in self.model.layers:
            query = linear(hidden, layer.wq)
            key = linear(hidden, layer.wk)
            value = linear(hidden, layer.wv)
            attention_input = mx.concatenate(
                (key, value, query[..., key.shape[-1] + value.shape[-1] :]), axis=-1
            )
            attention_output = linear(attention_input, layer.wo)
            mlp_input = hidden + attention_output
            gate = linear(mlp_input, layer.w_gate)
            up = linear(mlp_input, layer.w_up)
            mlp_output = linear(gate + up, layer.w_down)
            hidden = mlp_input + mlp_output
            outputs.extend((key, value))
        outputs.append(linear(hidden[:, -1:, :], self.model.lm_head))
        return outputs

    def attention(self):
        d = self.model.cfg["head_dim"]
        mask = "causal" if self.phase == "prefill" else None
        return [
            mx.fast.scaled_dot_product_attention(
                self.query, self.key, self.value, scale=d**-0.5, mask=mask
            )
            for _ in self.model.layers
        ]

    def pointwise(self):
        d = self.model.cfg["head_dim"]
        outputs = [self.model.embedding[self.tokens]]
        for layer in self.model.layers:
            outputs.extend(
                (
                    rms_norm(self.hidden, layer.ln1),
                    rms_norm(self.hidden, layer.ln2),
                    rms_norm(self.query_rows, layer.q_norm),
                    rms_norm(self.key_rows, layer.k_norm),
                )
            )
            offset = 0 if self.phase == "prefill" else self.rows
            outputs.append(
                mx.fast.rope(
                    self.query_rows,
                    d,
                    traditional=False,
                    base=10000.0,
                    scale=1.0,
                    offset=offset,
                )
            )
            outputs.append(
                mx.fast.rope(
                    self.key_rows,
                    d,
                    traditional=False,
                    base=10000.0,
                    scale=1.0,
                    offset=offset,
                )
            )
            outputs.append(silu(self.gate) * self.up)
            outputs.extend((self.hidden + self.hidden, self.hidden + self.hidden))
        outputs.append(rms_norm(self.hidden[:, -1:, :], self.model.norm))
        return outputs

    def cache(self):
        if self.phase == "prefill":
            return [self.key, self.value]
        previous_key = self.key[:, :, : -self.rows, :]
        previous_value = self.value[:, :, : -self.rows, :]
        new_key = self.key[:, :, -self.rows :, :]
        new_value = self.value[:, :, -self.rows :, :]
        outputs = []
        for _ in self.model.layers:
            outputs.append(mx.concatenate((previous_key, new_key), axis=2))
            outputs.append(mx.concatenate((previous_value, new_value), axis=2))
        return outputs

    def builders(self, include_cache=True):
        items = [
            ("projections", self.projections),
            ("attention", self.attention),
            ("pointwise", self.pointwise),
        ]
        if include_cache:
            items.append(("cache", self.cache))
        return tuple(items)


def benchmark_groups(
    builders, warmup, iterations, rotate=True, sync=True, position_log=None
):
    """Return {group: [seconds]}; mirrors the bench's rotation and warmup."""
    if rotate:
        orders = [builders[o:] + builders[:o] for o in range(len(builders))]
    else:
        orders = [builders]

    def measure(build):
        start = perf_counter()
        outputs = build()
        if sync:
            mx.eval(*outputs)
        return perf_counter() - start

    for round_index in range(warmup):
        for _, build in orders[round_index % len(orders)]:
            measure(build)
    samples = {name: [] for name, _ in builders}
    for round_index in range(iterations):
        order = orders[(warmup + round_index) % len(orders)]
        for position, (name, build) in enumerate(order):
            elapsed = measure(build)
            samples[name].append(elapsed)
            if position_log is not None:
                position_log.setdefault(name, {}).setdefault(position, []).append(
                    elapsed
                )
    return samples


def to_us(samples):
    return [value * 1e6 for value in samples]


def summarize(samples_us, names):
    medians = {name: statistics.median(samples_us[name]) for name in names}
    total = sum(medians.values())
    return medians, total


def print_table(title, headers, rows):
    print()
    print(title)
    widths = [
        max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def measure_floor():
    """Median wall time of one mx.eval carrying a single trivial op."""
    x = mx.random.normal((64,)).astype(DTYPE)
    mx.eval(x)

    def round_trip():
        mx.eval(x + x)

    for _ in range(20):
        round_trip()
    samples = []
    for _ in range(ITERATIONS):
        start = perf_counter()
        round_trip()
        samples.append((perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main():
    first_start = perf_counter()
    mx.eval(mx.random.normal((64, 64)).astype(DTYPE) + 1)
    first_eval = (perf_counter() - first_start) * 1e6
    floor = measure_floor()
    model = TinyModel(CFG)
    cfg = CFG
    print(
        f"mlx={mx.__version__} dtype=bfloat16 device={mx.default_device()} "
        f"one-eval floor={floor:.2f}us process-first-eval={first_eval:.2f}us"
    )
    print(
        f"model layers={cfg['layers']} E={cfg['hidden']} H={cfg['heads']} "
        f"G={cfg['kv_heads']} D={cfg['head_dim']} I={cfg['intermediate']} "
        f"vocab={cfg['vocab']}"
    )
    print(
        f"parameters={model.parameter_count} "
        f"weights={model.parameter_bytes / 1024:.1f} KiB"
    )

    # One real prefill of 8 tokens fills the cache the decode step reads.
    prompt = mx.array([[7, 11, 23, 42, 99, 128, 256, 401]], dtype=mx.int32)
    _, cache = model.forward(prompt, cache=None, offset=0)
    mx.eval(*[a for pair in cache for a in pair])
    offset = cfg["context"]

    # Dispatch counting pass: how many emit sites each group owns.
    counts = dict.fromkeys(GROUPS, 0)

    def count_emit(group, _value):
        counts[group] += 1

    model.forward(
        mx.array([[5]], dtype=mx.int32), cache=cache, offset=offset, emit=count_emit
    )
    total_sites = sum(counts.values())

    replay = KernelReplay(model, rows=1, context=offset + 1)
    builders = replay.builders()

    # [1] rotated, one eval per builder
    positions = {}
    rotated = benchmark_groups(
        builders, WARMUP, ITERATIONS, rotate=True, sync=True, position_log=positions
    )
    rotated_us = {name: to_us(samples) for name, samples in rotated.items()}
    medians, median_total = summarize(rotated_us, [n for n, _ in builders])

    # [2] same order, no mx.eval at all
    build_only = benchmark_groups(builders, WARMUP, ITERATIONS, rotate=True, sync=False)
    build_only_us = {name: to_us(samples) for name, samples in build_only.items()}
    build_medians, build_total = summarize(build_only_us, [n for n, _ in builders])

    # [3] fixed order, projections first, everything else unchanged
    fixed = benchmark_groups(builders, WARMUP, ITERATIONS, rotate=False, sync=True)
    fixed_us = {name: to_us(samples) for name, samples in fixed.items()}
    fixed_medians, fixed_total = summarize(fixed_us, [n for n, _ in builders])

    # A second identical rotated run measures run-to-run drift, the floor that
    # any order comparison has to beat.
    repeat = benchmark_groups(builders, WARMUP, ITERATIONS, rotate=True, sync=True)
    repeat_us = {name: to_us(samples) for name, samples in repeat.items()}
    repeat_medians, repeat_total = summarize(repeat_us, [n for n, _ in builders])

    print_table(
        f"[1] rotated replay, one mx.eval per builder "
        f"(warmup={WARMUP}, iterations={ITERATIONS})",
        ["group", "median_us", "share", "op sites", "site share", "median/floor"],
        [
            (
                name,
                f"{medians[name]:.2f}",
                f"{medians[name] / median_total:.1%}",
                counts[name],
                f"{counts[name] / total_sites:.1%}",
                f"{medians[name] / floor:.2f}x",
            )
            for name in GROUPS
        ]
        + [
            (
                "sum",
                f"{median_total:.2f}",
                "100.0%",
                total_sites,
                "100.0%",
                f"{median_total / floor:.2f}x",
            )
        ],
    )

    print_table(
        "[2] same groups, build() only, no mx.eval",
        ["group", "build_us", "build share", "eval_us", "eval share", "eval/build"],
        [
            (
                name,
                f"{build_medians[name]:.2f}",
                f"{build_medians[name] / build_total:.1%}",
                f"{medians[name]:.2f}",
                f"{medians[name] / median_total:.1%}",
                f"{medians[name] / build_medians[name]:.2f}x",
            )
            for name in GROUPS
        ]
        + [
            (
                "sum",
                f"{build_total:.2f}",
                "100.0%",
                f"{median_total:.2f}",
                "100.0%",
                f"{median_total / build_total:.2f}x",
            )
        ],
    )

    print_table(
        "[3b] median_us by execution position within the round (rotated run)",
        ["group", "pos 0", "pos 1", "pos 2", "pos 3", "max/min"],
        [
            (
                name,
                *[
                    f"{statistics.median(to_us(positions[name][p])):.1f}"
                    if p in positions[name]
                    else "-"
                    for p in range(len(builders))
                ],
                f"{max(statistics.median(to_us(positions[name][p])) for p in positions[name]) / min(statistics.median(to_us(positions[name][p])) for p in positions[name]):.2f}x",
            )
            for name in GROUPS
        ],
    )

    print_table(
        "[3] fixed order (projections first) vs rotated, against run-to-run drift",
        [
            "group",
            "rotated_us",
            "rotated repeat_us",
            "repeat/rotated",
            "fixed_us",
            "fixed/rotated",
        ],
        [
            (
                name,
                f"{medians[name]:.2f}",
                f"{repeat_medians[name]:.2f}",
                f"{repeat_medians[name] / medians[name]:.2f}x",
                f"{fixed_medians[name]:.2f}",
                f"{fixed_medians[name] / medians[name]:.2f}x",
            )
            for name in GROUPS
        ]
        + [
            (
                "sum",
                f"{median_total:.2f}",
                f"{repeat_total:.2f}",
                f"{repeat_total / median_total:.2f}x",
                f"{fixed_total:.2f}",
                f"{fixed_total / median_total:.2f}x",
            )
        ],
    )

    print_table(
        "[4] mean vs median, rotated run's samples",
        [
            "group",
            "median_us",
            "mean_us",
            "mean/median",
            "max_us",
            "max/median",
            "mean_wo_max",
        ],
        [
            (
                name,
                f"{statistics.median(rotated_us[name]):.2f}",
                f"{statistics.mean(rotated_us[name]):.2f}",
                f"{statistics.mean(rotated_us[name]) / statistics.median(rotated_us[name]):.2f}x",
                f"{max(rotated_us[name]):.2f}",
                f"{max(rotated_us[name]) / statistics.median(rotated_us[name]):.2f}x",
                f"{statistics.mean(sorted(rotated_us[name])[:-1]):.2f}",
            )
            for name in GROUPS
        ]
        + [
            (
                "sum",
                f"{median_total:.2f}",
                f"{sum(statistics.mean(rotated_us[n]) for n in GROUPS):.2f}",
                f"{sum(statistics.mean(rotated_us[n]) for n in GROUPS) / median_total:.2f}x",
                "",
                "",
                "",
            )
        ],
    )

    # [5] real decode step: one eval, everything materialized.
    token = mx.array([[503]], dtype=mx.int32)

    def real_step():
        logits, new_cache = model.forward(token, cache=cache, offset=offset)
        mx.eval(logits, *[a for pair in new_cache for a in pair])

    for _ in range(WARMUP):
        real_step()
    real_samples = []
    for _ in range(ITERATIONS):
        start = perf_counter()
        real_step()
        real_samples.append((perf_counter() - start) * 1e6)
    real_median = statistics.median(real_samples)
    real_mean = statistics.mean(real_samples)
    recomputed = sum(statistics.mean(rotated_us[n]) for n in GROUPS)

    print_table(
        "[5] real decode step vs replay sum",
        ["quantity", "us", "vs real median"],
        [
            ("real step, median", f"{real_median:.2f}", "1.00x"),
            ("real step, mean", f"{real_mean:.2f}", f"{real_mean / real_median:.2f}x"),
            (
                "replay sum of medians",
                f"{median_total:.2f}",
                f"{median_total / real_median:.2f}x",
            ),
            (
                "replay sum of means",
                f"{recomputed:.2f}",
                f"{recomputed / real_median:.2f}x",
            ),
            (
                "replay build-only sum",
                f"{build_total:.2f}",
                f"{build_total / real_median:.2f}x",
            ),
            (
                "4 x one-eval floor",
                f"{4 * floor:.2f}",
                f"{4 * floor / real_median:.2f}x",
            ),
        ],
    )

    # [6] real forward, sync after every stage: ordering reference with a known bias.
    class StageTimer:
        def __init__(self):
            self.buckets = dict.fromkeys(GROUPS, 0.0)
            self.syncs = 0
            self._last = perf_counter()

        def mark(self, group, value):
            mx.eval(value)
            now = perf_counter()
            self.buckets[group] += (now - self._last) * 1e6
            self._last = now
            self.syncs += 1

    def staged_step():
        timer = StageTimer()
        logits, new_cache = model.forward(
            token, cache=cache, offset=offset, emit=timer.mark
        )
        mx.eval(logits, *[a for pair in new_cache for a in pair])
        return timer

    for _ in range(WARMUP):
        staged_step()
    staged = {name: [] for name in GROUPS}
    sync_counts = []
    for _ in range(ITERATIONS):
        timer = staged_step()
        sync_counts.append(timer.syncs)
        for name in GROUPS:
            staged[name].append(timer.buckets[name])
    staged_medians = {name: statistics.median(staged[name]) for name in GROUPS}
    staged_total = sum(staged_medians.values())
    syncs = int(statistics.median(sync_counts))

    print_table(
        f"[6] real forward, sync after every stage ({syncs} syncs/step)",
        [
            "group",
            "staged_us",
            "staged share",
            "replay_us",
            "replay share",
            "staged/replay",
        ],
        [
            (
                name,
                f"{staged_medians[name]:.2f}",
                f"{staged_medians[name] / staged_total:.1%}",
                f"{medians[name]:.2f}",
                f"{medians[name] / median_total:.1%}",
                f"{staged_medians[name] / medians[name]:.2f}x",
            )
            for name in GROUPS
        ]
        + [
            (
                "sum",
                f"{staged_total:.2f}",
                "100.0%",
                f"{median_total:.2f}",
                "100.0%",
                f"{staged_total / median_total:.2f}x",
            )
        ],
    )
    print(
        f"staged total / real single-eval median = {staged_total / real_median:.2f}x; "
        f"mean cost per stage = {staged_total / syncs:.2f}us vs one-eval floor {floor:.2f}us "
        f"({staged_total / syncs / floor:.2f}x)"
    )

    # Prefill, same four groups, for the phase comparison.
    prefill_replay = KernelReplay(model, rows=cfg["context"], context=cfg["context"])
    prefill_builders = prefill_replay.builders(include_cache=False)
    prefill = benchmark_groups(
        prefill_builders, WARMUP, ITERATIONS, rotate=True, sync=True
    )
    prefill_us = {name: to_us(samples) for name, samples in prefill.items()}
    prefill_medians, prefill_total = summarize(
        prefill_us, [name for name, _ in prefill_builders]
    )

    def prefill_step():
        _, new_cache = model.forward(prompt, cache=None, offset=0)
        mx.eval(*[a for pair in new_cache for a in pair])

    prefill_samples = []
    for _ in range(WARMUP):
        prefill_step()
    for _ in range(ITERATIONS):
        start = perf_counter()
        prefill_step()
        prefill_samples.append((perf_counter() - start) * 1e6)
    prefill_real = statistics.median(prefill_samples)

    print_table(
        "[7] prefill (L=8) replay vs real single-eval step",
        ["group", "replay_us", "replay share"],
        [
            (
                name,
                f"{prefill_medians[name]:.2f}",
                f"{prefill_medians[name] / prefill_total:.1%}",
            )
            for name, _ in prefill_builders
        ]
        + [("sum", f"{prefill_total:.2f}", "100.0%")],
    )
    print(
        f"prefill real median = {prefill_real:.2f} us, "
        f"replay sum / real = {prefill_total / prefill_real:.2f}x"
    )

    report_sweep(floor)


def report_sweep(floor):
    rows = []
    for hidden in SWEEP:
        cfg = dict(CFG)
        cfg["hidden"] = hidden
        cfg["head_dim"] = hidden // cfg["heads"]
        cfg["intermediate"] = hidden * 2
        model = TinyModel(cfg)
        prompt, cache = (
            mx.array([[7, 11, 23, 42, 99, 128, 256, 401]], dtype=mx.int32),
            None,
        )
        _, cache = model.forward(prompt, cache=None, offset=0)
        mx.eval(*[a for pair in cache for a in pair])

        replay = KernelReplay(model, rows=1, context=cfg["context"] + 1)
        builders = replay.builders()
        samples = benchmark_groups(builders, 3, 31, rotate=True, sync=True)
        medians = {
            name: statistics.median(to_us(values)) for name, values in samples.items()
        }
        total = sum(medians[name] for name, _ in builders)

        token = mx.array([[503]], dtype=mx.int32)

        def step(current_model):
            logits, new_cache = current_model.forward(
                token, cache=cache, offset=cfg["context"]
            )
            mx.eval(logits, *[a for pair in new_cache for a in pair])

        for _ in range(3):
            step(model)
        real_samples = []
        for _ in range(31):
            start = perf_counter()
            step(model)
            real_samples.append((perf_counter() - start) * 1e6)
        real = statistics.median(real_samples)

        rows.append(
            (
                hidden,
                f"{model.parameter_bytes / 1024 / 1024:.2f}",
                f"{total:.2f}",
                f"{real:.2f}",
                f"{total / real:.2f}x",
                f"{4 * floor / real:.1%}",
                f"{medians['projections'] / total:.1%}",
                f"{medians['attention'] / total:.1%}",
                f"{medians['pointwise'] / total:.1%}",
                f"{medians['cache'] / total:.1%}",
            )
        )
        del model

    print_table(
        "[8] same harness across hidden sizes (decode, warmup=3, iterations=31)",
        [
            "E",
            "MiB",
            "replay_us",
            "real_us",
            "replay/real",
            "4xfloor/real",
            "proj",
            "attn",
            "point",
            "cache",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
