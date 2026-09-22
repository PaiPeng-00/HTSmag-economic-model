import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(root: Path) -> Optional[str]:
    git_dir: Optional[Path] = None
    for candidate_root in (root, *root.parents):
        candidate = candidate_root / ".git"
        if candidate.is_dir():
            git_dir = candidate
            break
    if git_dir is None:
        return None
    head = git_dir / "HEAD"
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref = content.split(":", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                value, packed_ref = line.split(" ", 1)
                if packed_ref == ref:
                    return value
        return None
    return content or None


def _iter_files(root: Path, patterns: Optional[Iterable[str]] = None) -> Iterable[Path]:
    if patterns is None:
        yield from root.rglob("*")
        return
    for pattern in patterns:
        yield from root.rglob(pattern)


def write_manifest(
    repo_root: Path,
    config_path: Path,
    inputs_root: Path,
    outputs_root: Path,
    manifest_path: Path,
    metadata: Optional[Mapping[str, Any]] = None,
    output_paths: Optional[Iterable[Path]] = None,
) -> None:
    git_hash = _git_head(repo_root)
    config_hash = _sha256_file(config_path) if config_path.exists() else None

    input_files = []
    if inputs_root.exists():
        for path in inputs_root.rglob("*"):
            if path.is_file():
                input_files.append({
                    "path": str(path.relative_to(repo_root)),
                    "sha256": _sha256_file(path),
                })

    output_files = []
    selected_outputs = (
        [Path(path) for path in output_paths]
        if output_paths is not None
        else [path for path in outputs_root.rglob("*") if path.is_file()]
    )
    for path in selected_outputs:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Manifest output is missing: {path}")
        output_files.append({
            "path": str(path.relative_to(repo_root)),
            "sha256": _sha256_file(path),
        })

    payload = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash or "unknown",
        "code_version": git_hash or "unknown",
        "config_path": str(config_path.relative_to(repo_root)) if config_path.exists() else None,
        "config_sha256": config_hash,
        "inputs_root": str(inputs_root.relative_to(repo_root)) if inputs_root.exists() else None,
        "inputs": input_files,
        "outputs_root": str(outputs_root.relative_to(repo_root)) if outputs_root.exists() else None,
        "outputs": output_files,
    }
    if metadata:
        protected = set(payload).intersection(metadata)
        if protected:
            raise ValueError(f"Manifest metadata overwrites protected keys: {sorted(protected)}")
        payload.update(dict(metadata))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
