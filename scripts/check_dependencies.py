"""Environment check: verifies every package the system needs imports cleanly.

Run:  python scripts/check_dependencies.py
Exits 0 when all imports succeed, 1 with the missing list otherwise.
"""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_MODULES = [
    "langgraph",
    "langchain",
    "langchain_google_genai",
    "langchain_ollama",
    "llama_index",
    "llama_index.embeddings.ollama",
    "llama_index.vector_stores.chroma",
    "chromadb",
    "crawl4ai",
    "docx2txt",
    "pypdf",
    "requests",
    "dotenv",  # provided by python-dotenv
    "docx",    # provided by python-docx
]


def main() -> int:
    missing = []
    for name in _MODULES:
        try:
            importlib.import_module(name)
            print(f"OK   {name}")
        except Exception as exc:  # noqa: BLE001 - diagnostic must not crash
            missing.append(name)
            print(f"FAIL {name} -> {type(exc).__name__}: {str(exc)[:120]}")
    if missing:
        print(f"\nMISSING: {missing}")
        print("Fix with: pip install -r requirements.txt")
        return 1
    print("\nALL_DEPENDENCIES_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
