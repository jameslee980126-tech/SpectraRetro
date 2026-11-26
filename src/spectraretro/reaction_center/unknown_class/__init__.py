"""Import the unknown-class reaction-center model through its original Python name."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "unknown-class" / "reaction_center"))
