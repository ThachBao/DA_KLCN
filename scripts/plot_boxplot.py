
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="dice", choices=["dice","iou","acc","time"])
    ap.add_argument("--out", default="results/boxplot.png")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv, encoding="utf-8"))]
    by_algo = {}
    for r in rows:
        by_algo.setdefault(r["algo"], []).append(float(r[args.metric]))
    labels = sorted(by_algo.keys())
    data = [by_algo[k] for k in labels]
    plt.figure()
    plt.boxplot(data)
    plt.xticks(range(1, len(labels)+1), labels, rotation=0)
    plt.ylabel(args.metric.upper())
    plt.title(f"Boxplot of {args.metric.upper()} by Algorithm")
    plt.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True); plt.savefig(args.out, dpi=150)
    print("Saved figure ->", args.out)

if __name__ == "__main__":
    main()
