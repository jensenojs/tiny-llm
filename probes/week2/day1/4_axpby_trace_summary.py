"""Summarize one three-layout axpby run exported by xctrace.

Inputs are XML tables from the same run: metal-application-encoders-list,
metal-gpu-intervals, metal-gpu-counter-intervals, and gpu-counter-info.
Each layout performs 3 warmups and 30 timed calls. The script verifies the
expected 99 encoders and excludes each layout's first three encoders.
Counter intervals describe the whole GPU; integration clips each sample to
the target process's observed GPU execution intervals.
"""

import argparse
import bisect
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def rows(path):
    """Resolve xctrace's document-local id/ref values while streaming rows."""
    refs = {}
    scalar_tags = {
        "start-time", "duration", "uint32", "uint64", "fixed-decimal",
        "percent", "boolean", "metal-command-buffer-id", "process",
        "gpu-counter-name", "string",
    }
    parent = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "node":
            parent = element
        if event != "end" or element.tag != "row":
            continue
        for child in element.iter():
            if "id" in child.attrib and child.tag in scalar_tags:
                refs[child.attrib["id"]] = (
                    child.text, child.attrib.get("fmt", child.text)
                )
        values = []
        for child in element:
            if "ref" in child.attrib:
                values.append(refs.get(child.attrib["ref"], (None, None)))
            else:
                values.append((child.text, child.attrib.get("fmt", child.text)))
        yield values
        parent.remove(element)
        element.clear()


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def summarize(app, gpu, counters, info, samples=None):
    metadata = {}
    for row in rows(info):
        metadata[int(row[1][0])] = {
            "name": row[2][0], "description": row[5][0], "type": row[7][0]
        }
    encoders = [(int(r[0][0]), r[3][1], int(r[12][0])) for r in rows(app)]
    encoders.sort()
    if len(encoders) != 99:
        raise ValueError(f"Expected 99 encoders (3 * (3+30)), got {len(encoders)}")
    processes = {r[1] for r in encoders}
    if len(processes) != 1:
        raise ValueError(f"Expected one target process, got {processes}")
    if len({r[2] for r in encoders}) != 99:
        raise ValueError("Encoder IDs must be unique")
    names = ["contiguous", "transpose", "strided_rows"]
    assignments = {}
    for group, name in enumerate(names):
        for _, _, encoder_id in encoders[group * 33 + 3:(group + 1) * 33]:
            assignments[encoder_id] = name
    intervals = defaultdict(list)
    all_target = defaultdict(list)
    for row in rows(gpu):
        if row[10][1] not in processes:
            continue
        start, duration = int(row[0][0]), int(row[1][0])
        encoder_id = int(row[16][0])
        all_target[encoder_id].append([start, start + duration])
        if encoder_id in assignments:
            intervals[assignments[encoder_id]].append([start, start + duration])
    for encoder_id in assignments:
        if encoder_id not in all_target:
            raise ValueError(f"Missing GPU execution for encoder {encoder_id}")
    intervals = {name: merge(intervals[name]) for name in names}
    starts = {name: [a for a, _ in intervals[name]] for name in names}
    wanted = {3, 17, 21, 23, 24, 46, 68, 69, 70, 71, 72, 79, 80, 81, 82, 83}
    accum = {name: defaultdict(lambda: [0.0, 0, 0]) for name in names}
    writer = csv.writer(samples) if samples is not None else None
    if writer:
        writer.writerow(["layout", "counter_id", "sample_start_ns",
                         "sample_duration_ns", "value", "overlap_ns"])
    row_count = 0
    for row in rows(counters):
        row_count += 1
        counter_id = int(row[2][0])
        if counter_id not in wanted:
            continue
        start, duration = int(row[0][0]), int(row[1][0])
        end, value = start + duration, float(row[5][0])
        for name in names:
            spans = intervals[name]
            index = max(0, bisect.bisect_right(starts[name], start) - 1)
            overlap = 0
            while index < len(spans) and spans[index][0] < end:
                a, b = spans[index]
                overlap += max(0, min(end, b) - max(start, a))
                index += 1
            if overlap:
                if writer:
                    writer.writerow([name, counter_id, start, duration, value, overlap])
                entry = accum[name][counter_id]
                entry[0] += value * overlap
                entry[1] += overlap
                entry[2] += 1
    output = {
        "target_process": next(iter(processes)),
        "encoder_count": len(encoders),
        "counter_interval_rows": row_count,
        "scope": "Whole-GPU samples clipped to target GPU intervals; mixed-process traffic",
        "groups": {},
    }
    for name in names:
        spans = intervals[name]
        active_ns = sum(b - a for a, b in spans)
        result = {
            "calls": 30, "active_ms": active_ns / 1e6,
            "gpu_ms_per_call": active_ns / 30 / 1e6,
            "first_ns": spans[0][0], "last_ns": spans[-1][1],
            "execution_spans_ns": spans,
            "counters": {},
        }
        for counter_id, (weighted, covered_ns, samples) in accum[name].items():
            meta = metadata[counter_id]
            metric = {
                **meta, "weighted_mean": weighted / covered_ns,
                "covered_fraction": covered_ns / active_ns,
                "overlapping_samples": samples,
            }
            if counter_id in {70, 71, 72, 83}:
                metric["integrated_MB_per_call"] = weighted / 1e9 * 1000 / 30
            result["counters"][str(counter_id)] = metric
        output["groups"][name] = result
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("app", "gpu", "counters", "info", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--samples", type=Path, help="Write overlapping samples as CSV")
    args = parser.parse_args()
    if args.samples:
        with args.samples.open("w", newline="") as samples:
            result = summarize(args.app, args.gpu, args.counters, args.info, samples)
    else:
        result = summarize(args.app, args.gpu, args.counters, args.info)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, group in result["groups"].items():
        print(name, f'{group["gpu_ms_per_call"]:.4f} GPU ms/call')
        for metric in group["counters"].values():
            if metric["name"] in {
                "Buffer L1 Miss Rate", "GPU Read Bandwidth",
                "GPU Write Bandwidth", "Last Level Cache Bandwidth",
            }:
                print(" ", metric)
