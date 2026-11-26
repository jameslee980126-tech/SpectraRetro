"""Import this before vendor src; never resolve the mutable historical tree."""
from pathlib import Path
import sys
import inspect

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "orchestration"))
try:
    from local_runtime import ensure_runtime
except ImportError:
    ensure_runtime = lambda: None
ensure_runtime()
import torch

VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "stage2"
if not (VENDOR_ROOT / "src" / "edm.py").is_file():
    raise RuntimeError("The frozen vendor Stage2 source is absent")
if "src" in sys.modules:
    imported = Path(sys.modules["src"].__file__).resolve()
    if VENDOR_ROOT not in imported.parents:
        raise RuntimeError("Refusing a Stage2 src import from outside the isolated snapshot")
sys.path.insert(0, str(VENDOR_ROOT))

# Compatibility for the already installed historical PyTorch 1.11 environment.
# The cache/checkpoints in this isolated package are trusted local artifacts.
if "weights_only" not in inspect.signature(torch.load).parameters:
    _torch_load = torch.load

    def _load_legacy(*args, **kwargs):
        restricted = kwargs.pop("weights_only", False)
        if restricted:
            raise RuntimeError("weights_only=True is unsupported by this historical torch")
        return _torch_load(*args, **kwargs)

    torch.load = _load_legacy
