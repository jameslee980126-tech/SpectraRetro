"""One-command unknown-class research reproduction."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'src'))
from spectraretro.run import main

if __name__ == '__main__':
    main('unknown')
