from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

REGISTRY_FIELDS = (
    "stage", "dataset_id", "schema_version", "path", "format", "rows", "columns",
    "primary_key", "scope", "units", "source_SHA", "output_SHA", "source_commit",
    "matrix_SHA", "status",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def registry_record(*, stage: str, dataset_id: str, schema_version: str, path: Path,
                    fmt: str, rows: int | None, columns: Iterable[str], primary_key: str,
                    scope: str, units: str, source_sha: str, source_commit: str,
                    matrix_sha: str, status: str = "PASS") -> dict[str, Any]:
    record = {
        "stage": stage, "dataset_id": dataset_id, "schema_version": schema_version,
        "path": str(path), "format": fmt, "rows": rows, "columns": list(columns),
        "primary_key": primary_key, "scope": scope, "units": units,
        "source_SHA": source_sha, "output_SHA": sha256_file(path),
        "source_commit": source_commit, "matrix_SHA": matrix_sha, "status": status,
    }
    missing = [field for field in REGISTRY_FIELDS if field not in record]
    if missing:
        raise ValueError(f"publication registry fields missing: {missing}")
    return record
