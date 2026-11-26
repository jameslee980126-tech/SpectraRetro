#!/usr/bin/env python3
import argparse
import runpy
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    args, runner_args = parser.parse_known_args()

    original_div = torch.div

    def compatible_div(input_tensor, other, *, rounding_mode=None, out=None):
        if (
            out is not None
            and not out.dtype.is_floating_point
            and not out.dtype.is_complex
            and rounding_mode is None
        ):
            rounding_mode = "floor"
        return original_div(
            input_tensor,
            other,
            rounding_mode=rounding_mode,
            out=out,
        )

    torch.div = compatible_div
    sys.argv = [str(args.runner), *runner_args]
    runpy.run_path(str(args.runner), run_name="__main__")


if __name__ == "__main__":
    main()
