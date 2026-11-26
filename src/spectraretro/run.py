"""Run reaction-center prediction, conditional 3D synthon completion and precursor ranking for the CHRetro benchmark."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectraretro.pipeline.runner import ROOT, run


def main(mode=None):
    """Run the model with an explicit class mode or a fixed benchmark wrapper."""
    parser = argparse.ArgumentParser(
        description='Run Stage 1 reaction centers, Stage 2 completion and decoding, '
                    'then fusion ranking of complete precursor candidates.')
    if mode is None:
        parser.add_argument('--mode', choices=['known', 'unknown'], required=True,
                            help='Supply (known) or withhold (unknown) the reaction-class label')
    else:
        parser.set_defaults(mode=mode)
    parser.add_argument('command', nargs='?', default='run', choices=['run', 'train', 'check', 'reproduce'],
                        help='run: complete pretrained evaluation; train: train both predictors then complete evaluation; check: inspect the complete plan; reproduce: sum saved result records')
    parser.add_argument('--config', type=Path, help='Repository-relative or absolute full workflow configuration')
    parser.add_argument('--output', type=Path, help='New directory for the entire run')
    for name in ['center', 'model', 'decode', 'score']:
        parser.add_argument('--' + name + '-python', help='Python executable; defaults to pipeline.python or the launcher environment')
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], help='Default CUDA; smoke runs default to CPU')
    parser.add_argument('--workers', type=int, default=2, help='Coordinate decoding processes')
    parser.add_argument('--check-only', action='store_true', help='Inspect inputs and automatic handoff without executing models')
    parser.add_argument('--smoke', action='store_true', help='Diagnostic only: two complete reactions, two draws and two diffusion intervals')
    parser.add_argument('--check', action='store_true', help='Check saved metrics with the reproduce command')
    args = parser.parse_args()
    mode = args.mode
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.command == 'reproduce':
        import runpy
        sys.argv = [str(ROOT / 'scripts/reproduce_results.py'), '--mode', mode] + (['--check'] if args.check else [])
        runpy.run_path(str(ROOT / 'scripts/reproduce_results.py'), run_name='__main__')
        return
    try:
        run(mode, args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
