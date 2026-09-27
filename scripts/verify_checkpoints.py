#!/usr/bin/env python3
"""Verify checkpoint hashes and optionally strict-load every model on CPU."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--load",action="store_true")
    args=p.parse_args()
    manifest=json.loads((args.root/"checkpoints/manifest.json").read_text())
    for task,entry in manifest["models"].items():
        directory=args.root/entry["directory"]
        digest=hashlib.sha256((directory/"weights.pt").read_bytes()).hexdigest()
        if digest!=entry["weights_sha256"]:raise RuntimeError(f"Hash mismatch: {task}")
        if args.load:
            from phylatent.models.loading import load_model
            load_model(directory,"cpu")
        print(f"{task}: SHA256 OK"+("; strict CPU load OK" if args.load else ""))

if __name__ == "__main__":
    main()
