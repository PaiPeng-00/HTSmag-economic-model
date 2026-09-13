#!/usr/bin/env python3
"""Copy the matching Zenodo payload into a clean repository clone."""
from pathlib import Path
import argparse, hashlib, shutil

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument("zenodo_data",type=Path);p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
payload=a.zenodo_data.resolve()/"payload";repo=a.repo.resolve()
if not payload.is_dir(): raise SystemExit("STOP: Zenodo payload not found")
for source in sorted(payload.rglob("*")):
    if not source.is_file(): continue
    target=repo/source.relative_to(payload)
    if target.exists(): raise SystemExit(f"STOP: target exists: {target}")
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    if sha(source)!=sha(target): raise SystemExit(f"STOP: hash mismatch: {target}")
print("ZENODO_HYDRATION=PASS")
