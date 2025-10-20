
# Convenience: run on BSDS300 test images (metrics skipped if masks absent)
# Usage:
#   python -m src.cli.quick_bsds300 --iters 50 --pop 20
import os, argparse, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--algos", default="mfwoa,woa,pso,ga,otsu")
    ap.add_argument("--Ks", default="2,3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="/mnt/data/results_bsds300_test")
    args = ap.parse_args()

    root = "/mnt/data/BSDS300/BSDS300/images/test"
    if not os.path.exists(root):
        print("BSDS300 test folder not found:", root)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "src.cli.run_experiment",
        "--dataset_root", root,
        "--out", args.out,
        "--algos", args.algos,
        "--Ks", args.Ks,
        "--iters", str(args.iters),
        "--pop", str(args.pop),
        "--runs", str(args.runs),
        "--seed", str(args.seed),
    ]
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
