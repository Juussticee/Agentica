"""Minimal ERP action logging — operational metadata only, never data.

One line per adapter operation:

    <ISO timestamp> | op=<operation> | adapter=<adapter> | status=success|failure
                    | duration_ms=<n> | records=<n> (success)
                    | error_category=<category> (failure)

NEVER written: passwords, API keys, access tokens, authorization headers,
query text, customer data, or record contents/IDs. The log lives at
``logs/erp_actions.log`` (git-ignored, path defined once in app/config.py).
Every write is fail-soft: observability must never break an ERP answer.
"""

from datetime import datetime
from typing import Optional

from app.config import ERP_LOG_PATH


def log_operation(
    operation: str,
    adapter: str,
    duration_ms: int,
    status: str,
    records: Optional[int] = None,
    error_category: Optional[str] = None,
) -> None:
    """Append one minimal log line; never raises."""
    parts = [
        datetime.now().astimezone().isoformat(timespec="seconds"),
        f"op={operation}",
        f"adapter={adapter}",
        f"status={status}",
        f"duration_ms={duration_ms}",
    ]
    if records is not None:
        parts.append(f"records={records}")
    if error_category is not None:
        parts.append(f"error_category={error_category}")
    line = " | ".join(parts) + "\n"
    try:
        ERP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ERP_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass  # logging must never break the ERP answer
