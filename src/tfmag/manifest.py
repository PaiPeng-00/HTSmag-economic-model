import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(root: Path) -> Optional[str]:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref = content.split(":", 1)[1].strip()
        ref_path = root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
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
    if outputs_root.exists():
        for path in outputs_root.rglob("*"):
            if path.is_file():
                output_files.append({
                    "path": str(path.relative_to(repo_root)),
                    "sha256": _sha256_file(path),
                })

    payload = {
        "code_version": git_hash or "unknown",
        "config_path": str(config_path.relative_to(repo_root)) if config_path.exists() else None,
        "config_sha256": config_hash,
        "inputs_root": str(inputs_root.relative_to(repo_root)) if inputs_root.exists() else None,
        "inputs": input_files,
        "outputs_root": str(outputs_root.relative_to(repo_root)) if outputs_root.exists() else None,
        "outputs": output_files,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
