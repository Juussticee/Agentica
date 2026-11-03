"""Structured-data specialists (2026-10-30).

Two layers, deliberately separated:

  * `dataset` — READER: turns one dataset file inside the configured workspace
    into typed columns and bounded rows. No model, no numbers guessed.
  * `analysis` — COMPUTATION: a small closed set of operations whose results
    are computed here, in Python, over the parsed rows. Every figure in a
    Data Analyst answer comes from this module.

Nothing in here writes a file, runs a shell command or reaches the network.
The model may only CHOOSE which operations to run (`app/agents/data_agent.py`)
and explain their results; it never supplies a number.
"""

from app.data.analysis import AnalysisError, OPS, run_step, run_steps
from app.data.dataset import (
    DATA_EXTENSIONS,
    Column,
    Dataset,
    DatasetError,
    load_dataset,
)

__all__ = [
    "AnalysisError",
    "DATA_EXTENSIONS",
    "OPS",
    "Column",
    "Dataset",
    "DatasetError",
    "load_dataset",
    "run_step",
    "run_steps",
]
