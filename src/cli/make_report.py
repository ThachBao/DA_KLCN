# Tạo báo cáo Markdown từ metrics/summary đã sinh
# Usage:
#   python -m src.cli.make_report --in "results" --out "results/report.md"
import os, csv, argparse, glob

def read_first(globpat):
    xs = sorted(glob.glob(globpat))
    return xs[0] if xs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Folder chứa results")
    ap.add_argument("--out", dest="outfile", required=True, help="Đường dẫn report .md")
    args = ap.parse_args()

    metrics = read_first(os.path.join(args.indir, "metrics_*.csv"))
    summary = read_first(os.path.join(args.indir, "summary_*.csv"))
    fig_fe = read_first(os.path.join(args.indir, "summary_FE_*.png"))
    fig_dice = read_first(os.path.join(args.indir, "summary_Dice_*.png"))

    lines = []
    lines.append("# Báo cáo thực nghiệm phân đoạn (MFWOA vs Otsu/PSO/GA/WOA)")
    lines.append("")
    lines.append("## 1. Cấu hình chạy")
    lines.append(f"- Thư mục kết quả: `{args.indir}`")
    lines.append("")
    lines.append("## 2. Bảng kết quả tổng hợp")
    if summary:
        lines.append(f"- File tổng hợp: `{os.path.basename(summary)}`")
        lines.append("")
        # Chèn bảng tóm tắt (hiển thị 10 dòng đầu)
        lines.append("| algo_K | mean_dice | mean_iou | mean_FE | mean_sec | n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        with open(summary, "r", newline="") as f:
            rd = csv.DictReader(f)
            for i, row in enumerate(rd):
                if i >= 10: break
                lines.append(f"| {row['algo_K']} | {row['mean_dice']} | {row['mean_iou']} | {row['mean_FE']} | {row['mean_sec']} | {row['n']} |")
    else:
        lines.append("- (Chưa có summary_*.csv. Hãy chạy với `--summary`)")

    lines.append("")
    lines.append("## 3. Biểu đồ so sánh")
    if fig_fe:
        lines.append(f"![Mean Fuzzy Entropy]({os.path.basename(fig_fe)})")
    if fig_dice:
        lines.append(f"![Mean Dice]({os.path.basename(fig_dice)})")
    if not (fig_fe or fig_dice):
        lines.append("- (Chưa có hình. Hãy chạy `--summary` để vẽ)")

    # Sigtest
    sig = read_first(os.path.join(args.indir, "sigtest_*.csv"))
    lines.append("")
    lines.append("## 4. Kiểm định ý nghĩa thống kê (Wilcoxon)")
    if sig:
        lines.append(f"- File: `{os.path.basename(sig)}`")
        lines.append("")
        lines.append("| metric | K | mfwoa_vs | n_pairs | W_stat | p_value |")
        lines.append("|---|---:|---|---:|---:|---:|")
        with open(sig, "r", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                lines.append(f"| {row['metric']} | {row['K']} | {row['mfwoa_vs']} | {row['n_pairs']} | {row['W_stat']} | {row['p_value']} |")
        lines.append("")
        lines.append("> *Gợi ý diễn giải:* p-value nhỏ (< 0.05) với `alternative='greater'` cho thấy MFWOA **tốt hơn có ý nghĩa thống kê** so với thuật toán baseline trên metric & K tương ứng.")
    else:
        lines.append("- (Chưa có sigtest. Hãy chạy với `--sigtest`)")

    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Report saved to:", args.outfile)

if __name__ == "__main__":
    main()
