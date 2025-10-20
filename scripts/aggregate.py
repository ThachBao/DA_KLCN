
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["algo"], int(r["K"]))].append(r)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["algo","K","n","dice_mean","dice_std","iou_mean","iou_std","time_mean","time_std"])
        for (algo,K), rs in sorted(buckets.items()):
            dice = np.array([float(r["dice"]) for r in rs])
            iou = np.array([float(r["iou"]) for r in rs])
            t = np.array([float(r["time"]) for r in rs])
            wr.writerow([algo, K, len(rs), dice.mean(), dice.std(), iou.mean(), iou.std(), t.mean(), t.std()])
    print(f"Saved summary -> {out}")

if __name__ == "__main__":
    main()
