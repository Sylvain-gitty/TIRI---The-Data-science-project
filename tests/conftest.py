"""Put `scripts/` on the path for the test suite.

Same mechanism the scripts and notebooks already use (`sys.path.insert` on the scripts
directory) rather than making `scripts/` an installable package — converting it would touch
the import preamble of ~40 scripts and every notebook, which is a large change for no gain
to the tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
