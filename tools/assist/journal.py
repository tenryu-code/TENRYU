"""Append-only JSONL journal utilities for assistant activity."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union


def sha256_hex(data: Union[str, bytes]) -> str:
    """Return the SHA-256 digest of text or bytes as hexadecimal."""
    import hashlib

    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> str:
    """Serialize an object using the journal's canonical JSON form."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _utc_clock() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JournalWriter:
    """Append records to a JSONL journal."""

    def __init__(
        self, path: str, clock: Optional[Callable[[], str]] = None
    ) -> None:
        self.path = path
        self.clock = clock if clock is not None else _utc_clock

    def append(self, kind: str, payload: dict) -> dict:
        record = {
            "schema": "tenryu.assist.journal.v0",
            "ts": self.clock(),
            "kind": kind,
            "payload": payload,
        }
        record["record_sha256"] = sha256_hex(canonical_json(record))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
        return record
