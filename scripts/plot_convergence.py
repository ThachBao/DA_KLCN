
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_glob", required=True)
    ap.add_argument("--out", default="results/convergence.png")
    args = ap.parse_args()

    paths = list(Path().glob(args.logs_glob))
    if not paths: raise SystemExit("No *_conv.json found")
    plt.figure()
    for p in paths:
        data = json.loads(p.read_text())
        y = data.get("best_fe", [])
        if not y: continue
        plt.plot(range(len(y)), y, alpha=0.6)
    plt.xlabel("Iteration"); plt.ylabel("Best Fuzzy Entropy"); plt.title("Convergence Curves")
    plt.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True); plt.savefig(args.out, dpi=150)
    print("Saved figure ->", args.out)

if __name__ == "__main__":
    main()
