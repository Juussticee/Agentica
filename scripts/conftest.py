"""pytest bootstrap: fail-closed Chroma production-store isolation.

pytest imports this conftest BEFORE any test module in this directory is
imported, so by the time project/test code can reach a construction site:

* the guard is armed process-wide (environment-based, inherited by child
  processes), and
* every ``chromadb`` client constructor is wrapped so ANY construction —
  however it is reached, including direct use inside a test or a module-level
  import-time initialization — is checked against the production store first.

See ``app/chroma_guard.py`` for the contract and ``docs/TESTING.md``
section F for what this does and does not guarantee.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()