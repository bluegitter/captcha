#!/usr/bin/env python3
"""Deterministically split labelled Java Kaptcha images into train/test sets."""
import argparse, random, shutil
from pathlib import Path

def main():
    p = argparse.ArgumentParser(); p.add_argument("source", type=Path); p.add_argument("output", type=Path); p.add_argument("--test-ratio", type=float, default=.2); p.add_argument("--seed", type=int, default=20261114); a = p.parse_args()
    files = sorted([*a.source.glob("*.jpg"), *a.source.glob("*.jpeg"), *a.source.glob("*.png")])
    if not files: raise ValueError(f"no images in {a.source}")
    rng = random.Random(a.seed); rng.shuffle(files); cut = int(len(files) * (1 - a.test_ratio))
    for name, group in (("train", files[:cut]), ("test", files[cut:])):
        target = a.output / name; target.mkdir(parents=True, exist_ok=True)
        for src in group: shutil.copy2(src, target / src.name)
        print(f"{name}: {len(group)}")
if __name__ == "__main__": main()
