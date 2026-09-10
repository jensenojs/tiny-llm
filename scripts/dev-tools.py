# =============================================================================
# 测试入口与 refsol 机制 — 学习注释
#
# `pdm run test` 经 pyproject.toml 的 [tool.pdm.scripts] (test.cmd) 转发到这里。
#
# 核心机制: tests_refsol/ 是测试的权威源, tests/ 是工作副本。
# `pdm run test --week X --day Y` 每次执行:
#   1) copy_test(force=True): 把 tests_refsol/test_week_X_day_Y.py 强制复制到
#      tests/, 覆盖你对测试文件的任何本地修改;
#   2) pytest -v tests/test_week_X_day_Y.py, 透传其余参数 (pytest 过滤器)。
#
# 含义: 你只能改 src/tiny_llm/ 下的实现来让测试通过, 不能改测试本身。
# 改测试 = 改课程权威, 只能手动改 tests_refsol/ 或 copy-test --force,
# 且会偏离 refsol 被 diff 发现。不带 --week/--day 时直接跑整个 tests/。
# `pdm run test-refsol` 直接在 tests_refsol/ 上跑, 用于对照参考实现通过率。
# =============================================================================

import argparse
import os
import shutil
from pathlib import Path

import pytest


def validate_week_day(args, required=False):
    week_provided = args.week is not None
    day_provided = args.day is not None
    if week_provided != day_provided or (required and not week_provided):
        print("Please provide both week and day")
        return False
    if week_provided and (args.week <= 0 or args.day <= 0):
        print("Week and day must be positive integers")
        return False
    return True


def test_source_file(args):
    return f"tests_refsol/test_week_{args.week}_day_{args.day}.py"


def copy_test(args, skip_if_exists=False, force=False):
    if not validate_week_day(args, required=True):
        return 1
    source_file = test_source_file(args)
    target_file = f"tests/test_week_{args.week}_day_{args.day}.py"
    if skip_if_exists and os.path.exists(target_file) and not force:
        # diff the two files and warn if they are different
        if Path(source_file).read_text() != Path(target_file).read_text():
            print(
                f"[WARNING] {target_file} already exists and is different from {source_file}"
            )
            print(
                f"You can run `pdm run copy-test --week {args.week} --day {args.day} --force` to update it"
            )
        return 0
    print(f"copying {source_file} to {target_file}")
    shutil.copyfile(source_file, target_file)
    return 0


def test(args):
    if not validate_week_day(args):
        return 1
    if args.week is not None:
        if args.week == 4:
            targets = []
            for day in range(1, args.day + 1):
                day_args = argparse.Namespace(
                    week=args.week,
                    day=day,
                )
                status = copy_test(day_args, force=True)
                if status:
                    return status
                targets.append(f"tests/test_week_{args.week}_day_{day}.py")
            return pytest.main(["-v", *targets] + args.remainders)
        copy_test(args, force=True)
        return pytest.main(
            ["-v", f"tests/test_week_{args.week}_day_{args.day}.py"] + args.remainders
        )
    return pytest.main(["-v", "tests"] + args.remainders)


def test_refsol(args):
    if not validate_week_day(args):
        return 1
    if args.week is not None:
        source_file = test_source_file(args)
        return pytest.main(["-v", source_file] + args.remainders)
    return pytest.main(["-v", "tests_refsol"] + args.remainders)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    copy_test_parser = subparsers.add_parser("copy-test")
    copy_test_parser.add_argument("--week", type=int, required=True)
    copy_test_parser.add_argument("--day", type=int, required=True)
    copy_test_parser.add_argument("--force", action="store_true")
    copy_test_parser.set_defaults(copy_test_parser=True)
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--week", type=int, required=False)
    test_parser.add_argument("--day", type=int, required=False)
    test_parser.add_argument("remainders", nargs="*")
    test_parser.set_defaults(test_parser=True)
    test_refsol_parser = subparsers.add_parser("test-refsol")
    test_refsol_parser.add_argument("--week", type=int, required=False)
    test_refsol_parser.add_argument("--day", type=int, required=False)
    test_refsol_parser.add_argument("remainders", nargs="*")
    test_refsol_parser.set_defaults(test_refsol_parser=True)
    args = parser.parse_args()
    if hasattr(args, "copy_test_parser"):
        return copy_test(args, force=args.force)
    if hasattr(args, "test_parser"):
        return test(args)
    if hasattr(args, "test_refsol_parser"):
        return test_refsol(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
