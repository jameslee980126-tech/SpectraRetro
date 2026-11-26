#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"expected one GLN evaluation marker, found {source.count(old)}")
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--phase", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args, runner_args = parser.parse_known_args()

    source = args.runner.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "        fname_pred = os.path.join(cmd_args.save_dir, 'test.pred')\n"
        "        eval_model('test', model, fname_pred)\n",
        f"        fname_pred = {str(args.output)!r}\n"
        f"        eval_model({args.phase!r}, model, fname_pred)\n",
    )

    code = compile(source, str(args.runner), "exec")
    if args.check_only:
        print(f"compatible: {args.runner}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime_root = args.runner.parents[2]
    os.chdir(runtime_root)
    sys.path.insert(0, str(runtime_root))
    sys.argv = [str(args.runner), *runner_args]
    namespace = {
        "__file__": str(args.runner),
        "__name__": "__main__",
    }
    exec(code, namespace)


if __name__ == "__main__":
    main()
