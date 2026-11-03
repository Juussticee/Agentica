"""Dataset reader for the Data Analyst (2026-10-30) — read-only, bounded.

One file, inside the SAME configured workspace boundary the coding capability
uses (`CODING_WORKSPACE_ROOT`) — this module adds no new filesystem
permission:

  * the path goes through `Workspace.resolve`, so `..`, absolute paths,
    symlink/junction escapes and credential-shaped names are refused by the
    existing boundary code, not a second copy of it;
  * only `.csv` / `.tsv` are datasets, and only those extensions are read here
    (the coding write path is untouched — a dataset is never patchable);
  * the file is size-capped BEFORE it is read, rows are capped while parsing,
    and a truncation is recorded so every later result can say what it was
    computed over;
  * values are kept as STRINGS; typing (`integer` / `number` / `text` /
    `empty`) is inferred from the whole loaded sample, so an operation can
    refuse a column that is not numeric instead of coercing it silently.

Deliberately absent: Excel/Parquet (no dependency, no binary guessing),
writing, sorting/joining outside the small analysis operation set.
"""

import csv
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.coding.workspace import Workspace, WorkspaceError

DATA_EXTENSIONS = frozenset({".csv", ".tsv"})

MAX_DATASET_BYTES = 8_000_000     # hard cap before reading
MAX_ROWS = 50_000                 # parsed row cap (head of the file)
MAX_COLUMNS = 200
MAX_HEADER_CHARS = 120

# Values treated as missing. Deliberately conservative: a real "-" or "0" is
# data, not a hole.
MISSING_VALUES = frozenset({"", "na", "n/a", "null", "none", "nan"})


class DatasetError(Exception):
    """The dataset cannot be read (boundary, format, size, shape)."""


@dataclass(frozen=True)
class Column:
    name: str
    kind: str            # "integer" | "number" | "text" | "empty"
    non_null: int
    missing: int
    distinct: int

    @property
    def numeric(self) -> bool:
        return self.kind in ("integer", "number")


@dataclass(frozen=True)
class Dataset:
    path: str
    columns: Tuple[Column, ...]
    rows: Tuple[Tuple[str, ...], ...]
    row_count: int
    truncated: bool
    delimiter: str

    def index(self, name: str) -> int:
        """Column position by exact (then case-insensitive) name."""
        for position, column in enumerate(self.columns):
            if column.name == name:
                return position
        lowered = str(name or "").strip().lower()
        for position, column in enumerate(self.columns):
            if column.name.lower() == lowered:
                return position
        raise DatasetError(f"no column named {name!r} in {self.path}")

    def column(self, name: str) -> Column:
        return self.columns[self.index(name)]

    def values(self, name: str) -> List[str]:
        position = self.index(name)
        return [row[position] for row in self.rows]

    def numbers(self, name: str) -> List[float]:
        """Numeric values of a column; refuses a non-numeric column."""
        column = self.column(name)
        if not column.numeric:
            raise DatasetError(
                f"column {column.name!r} holds {column.kind} values, not "
                "numbers, so it cannot be aggregated")
        out: List[float] = []
        for raw in self.values(name):
            if not _is_missing(raw):
                out.append(_to_number(raw))
        return out

    def schema_line(self) -> str:
        parts = []
        for column in self.columns:
            parts.append(
                f"{column.name} ({column.kind}, {column.non_null} non-null, "
                f"{column.missing} missing)")
        return "; ".join(parts)

    def coverage_note(self) -> str:
        """What the numbers were computed over — never omitted."""
        if self.truncated:
            return (f"computed over the first {self.row_count} rows of "
                    f"{self.path} (the file has more; rows were capped)")
        return f"computed over all {self.row_count} rows of {self.path}"


def _is_missing(value: object) -> bool:
    return str(value if value is not None else "").strip().lower() in MISSING_VALUES


def _to_number(value: str) -> float:
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        raise DatasetError(f"value {value!r} is not a number")


def _as_int_if_whole(value: str) -> Optional[int]:
    text = str(value).replace(",", "").strip()
    try:
        return int(text)
    except ValueError:
        return None


def _infer_kind(values: Sequence[str]) -> str:
    present = [v for v in values if not _is_missing(v)]
    if not present:
        return "empty"
    if all(_as_int_if_whole(v) is not None for v in present):
        return "integer"
    numeric = 0
    for value in present:
        try:
            float(str(value).replace(",", "").strip())
            numeric += 1
        except ValueError:
            return "text"
    return "number" if numeric == len(present) else "text"


def _unique_names(header: Sequence[str]) -> List[str]:
    """Stable, unique, non-empty column names (never silently collides)."""
    names: List[str] = []
    seen: Dict[str, int] = {}
    for position, raw in enumerate(header):
        name = str(raw or "").strip()[:MAX_HEADER_CHARS] or f"column_{position + 1}"
        key = name.lower()
        if key in seen:
            seen[key] += 1
            name = f"{name} ({seen[key]})"
            key = name.lower()
        seen.setdefault(key, 1)
        names.append(name)
    return names


def load_dataset(workspace: Workspace, relative: object) -> Dataset:
    """Read one dataset inside the workspace boundary. Read-only."""
    raw_path = str(relative or "").strip()
    if not raw_path:
        raise DatasetError("a dataset path is required")
    suffix = "." + raw_path.rsplit(".", 1)[-1].lower() if "." in raw_path else ""
    if suffix not in DATA_EXTENSIONS:
        raise DatasetError(
            f"not a dataset file: {raw_path} (supported: "
            + ", ".join(sorted(DATA_EXTENSIONS)) + ")")
    try:
        path = workspace.resolve(raw_path)
    except WorkspaceError as exc:
        raise DatasetError(str(exc)) from exc
    if not path.is_file():
        raise DatasetError(f"not a file: {raw_path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DatasetError(f"could not stat {raw_path}: {exc}") from exc
    if size > MAX_DATASET_BYTES:
        raise DatasetError(
            f"{raw_path} is too large to analyse ({size} bytes > "
            f"{MAX_DATASET_BYTES})")
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise DatasetError(f"could not read {raw_path}: {exc}") from exc

    delimiter = "\t" if suffix == ".tsv" else _sniff(text)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        raise DatasetError(f"{raw_path} is empty (no header row)")
    if not header:
        raise DatasetError(f"{raw_path} has no header row")
    if len(header) > MAX_COLUMNS:
        raise DatasetError(
            f"{raw_path} has {len(header)} columns (max {MAX_COLUMNS})")
    names = _unique_names(header)

    rows: List[Tuple[str, ...]] = []
    truncated = False
    for record in reader:
        if len(rows) >= MAX_ROWS:
            truncated = True
            break
        if not record or all(not str(cell).strip() for cell in record):
            continue
        # Pad short rows, join overflowing ones into the last cell: a ragged
        # file is preserved as data, never silently dropped.
        cells = [str(cell).strip() for cell in record[:len(names)]]
        if len(record) > len(names):
            cells[-1] = delimiter.join(
                [cells[-1]] + [str(c).strip() for c in record[len(names):]])
        cells += [""] * (len(names) - len(cells))
        rows.append(tuple(cells))

    columns: Tuple[Column, ...] = tuple(
        Column(
            name=names[position],
            kind=(kind := _infer_kind([row[position] for row in rows])),
            non_null=sum(1 for row in rows if not _is_missing(row[position])),
            missing=sum(1 for row in rows if _is_missing(row[position])),
            distinct=len({
                row[position].strip().lower() for row in rows
                if not _is_missing(row[position])
            }),
        )
        for position in range(len(names))
    )
    return Dataset(
        path=str(relative).replace("\\", "/"),
        columns=columns,
        rows=tuple(rows),
        row_count=len(rows),
        truncated=truncated,
        delimiter=delimiter,
    )


def _sniff(text: str) -> str:
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","
