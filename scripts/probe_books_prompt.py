"""Diagnostic probe: capture prompts flowing through the get_llm() bridge.

Runs the real books agent, so it consumes ONE Gemini request. Use it to see
exactly what the LlamaIndex query engine sends to the shared get_llm()
factory (chunk context + the exact refusal instruction).

Run:  python scripts/probe_books_prompt.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.llm_client as llm_client_module
from app.llm_client import llm_text

CAPTURED = []
real_get_llm = llm_client_module.get_llm  # keep the true factory BEFORE patching


def capturing_llm():
    inner = real_get_llm()

    class Wrapper:
        def invoke(self, prompt):
            CAPTURED.append(str(prompt))
            return inner.invoke(prompt)

    return Wrapper()


llm_client_module.get_llm = capturing_llm

from app.agents.books_agent import books_agent  # noqa: E402

r = books_agent({"question": "How much delta-v does a Hohmann transfer need?"})
print("=== final answer ===")
print(r["answer"][:300])
print("\n=== captured prompts:", len(CAPTURED))
for i, p in enumerate(CAPTURED):
    print(f"\n--- prompt {i} (len {len(p)}) ---")
    print(p[:1500])
    print("... [tail] ...")
    print(p[-500:])
