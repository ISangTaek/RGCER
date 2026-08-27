"""Root test configuration.

``--basetemp=.tmp/pytest`` in pytest.ini pins pytest's scratch directory
inside the workspace (the machine's default %TEMP% path is not writable in
some environments); pytest requires the parent directory to pre-exist.
"""

from pathlib import Path

Path(__file__).resolve().parent.joinpath(".tmp", "pytest").mkdir(parents=True, exist_ok=True)
