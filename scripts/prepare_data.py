#!/usr/bin/env python3
"""Copy or decompress an official local HDF5 archive into the data directory."""
import argparse
from pathlib import Path
import shutil

FILES={"cube":"cube_single_expert.h5","tworoom":"tworoom.h5","reacher":"reacher.h5","pusht":"pusht_expert_train.h5"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=FILES)
    p.add_argument("--source",type=Path,required=True,help="Downloaded .h5 or .h5.zst file")
    p.add_argument("--data-root",type=Path,default=Path("data"))
    args=p.parse_args()
    if not args.source.is_file():raise FileNotFoundError(args.source)
    args.data_root.mkdir(parents=True,exist_ok=True)
    target=args.data_root/FILES[args.task]
    if target.exists():raise FileExistsError(f"Refusing to overwrite {target}")
    partial=target.with_suffix(".h5.partial")
    if partial.exists():raise FileExistsError(partial)
    try:
        if args.source.suffix==".zst":
            import zstandard
            with args.source.open("rb") as src,partial.open("wb") as dst:
                zstandard.ZstdDecompressor().copy_stream(src,dst)
        else:
            shutil.copyfile(args.source,partial)
        with partial.open("rb") as handle:
            if handle.read(8)!=b"\x89HDF\r\n\x1a\n":
                raise ValueError("Input is not a standard HDF5 file")
        partial.rename(target)
    except Exception:
        if partial.exists():partial.unlink()
        raise
    print(target)

if __name__=="__main__":
    main()
