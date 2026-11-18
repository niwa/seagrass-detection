"""Pytest configuration: put the flat `scripts/` modules on sys.path so tests
can `import utils`, `import training`, `import sampling`, etc. directly,
matching how the notebooks import them."""

import pathlib
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
