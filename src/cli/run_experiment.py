import os, csv, time, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm

from ..dataset import find_images, pair_masks, read_gray
from ..utils import ensure_dir, set_seed, save_gray, overlay_mask, hungarian_match
from ..metrics import dice_score, iou_score, psnr, ssim
from ..fuzzy_entropy import fuzzy_entropy_objective
from ..segmentation import apply_thresholds_to_image

from ..algorithms.woa import woa_optimize
from ..algorithms.mfwoa import mfwoa_optimize
from ..algorithms.pso import pso_optimize
from ..algorithms.ga import ga_optimize
from ..algorithms.otsu import multi_otsu_thresholds

def hist256(img):
    h, _ = np.histogram(img, bins=256, range=(0,256))
    p = h.astype(np.float64) / (h.sum() + 1e-12)
    return p

def run_single_algo_on_image(algo, K, img_gray, iters, pop, seed, save_curve=None, curve_key=None):
    hist = hist256(img_gray)
    obj = fuzzy_entropy_objective(hist, K=K, s=2.0)

    history = []
    def wrapped_obj(x):
        val = obj(x)
        history.append(val)
        return val

    if algo == "woa":
        use_obj = wrapped_obj if save_curve is not None else obj
        T, best = woa_optimize(use_obj, K=K, pop=pop, iters=iters, seed=seed)
    elif algo == "pso":
        use_obj = wrapped_obj if save_curve is not None else obj
        T, best = pso_optimize(use_obj, K=K, pop=pop, iters=iters, seed=seed)
    elif algo == "ga":
        use_obj = wrapped_obj if save_curve is not None else obj
        T, best = ga_optimize(use_obj, K=K, pop=pop, iters=iters, seed=seed)
    elif algo == "otsu":
        thresholds = multi_otsu_thresholds((hist * 1e6).astype(int), classes=K+1)
        T = np.array(thresholds, dtype=np.int32)
        best = 0.0
    else:
        raise ValueError("Unknown algo: %s" % algo)

    if save_curve is not None and len(history) > 0:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(history)
            plt.xlabel('Iteration'); plt.ylabel('Fuzzy Entropy'); plt.title(curve_key)
            plt.tight_layout(); plt.savefig(save_curve); plt.close()
        except Exception as e:
            print("[warn] cannot save curve:", e)
    return T, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--images_glob", default=None)
    ap.add_argument("--masks_glob", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--algos", default="mfwoa,woa,pso,ga,otsu")
    ap.add_argument("--Ks", default="2,3")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rmp", type=float, default=0.3)
    ap.add_argument("--curves", action="store_true", help="Save convergence curves")
    ap.add_argument("--summary", action="store_true", help="Write per-algo summary CSV and charts")
    ap.add_argument("--sigtest", action="store_true", help="Wilcoxon test MFWOA vs each baseline (Dice/IoU)")
    ap.add_argument("--debug_glob", action="store_true", help="Print debug info for file discovery")

    args = ap.parse_args()
    set_seed(args.seed)

    image_paths = find_images(args.dataset_root, args.images_glob, debug=args.debug_glob)
    if len(image_paths) == 0:
        print("[ERROR] Không tìm thấy ảnh. Kiểm tra:")
        print(" - --dataset_root có đúng không?")
        print(" - --images_glob có khớp (ví dụ images/**/*.jpg) chưa?")
        print(" - Thử tạm --images_glob \"**/*.*\" để kiểm tra.")
    pairs = pair_masks(image_paths, args.dataset_root, args.masks_glob)

    algos = [s.strip().lower() for s in args.algos.split(",") if s.strip()]
    Ks = [int(k) for k in args.Ks.split(",") if k.strip()]

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_root = os.path.join(args.out)
    ensure_dir(out_root)

    metrics_path = os.path.join(out_root, f"metrics_{ts}.csv")
    with open(metrics_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image","algo","K","run","FE","Dice","IoU","PSNR","SSIM","thresholds"])

    summary_rows = []

    # Preload images
    hists, imgs_gray = {}, {}
    for ip in tqdm(image_paths, desc="Loading images"):
        img = read_gray(ip)
        imgs_gray[ip] = img
        h = np.histogram(img, bins=256, range=(0,256))[0].astype(np.float64)
        hists[ip] = h / (h.sum() + 1e-12)

    for run in range(args.runs):
        run_seed = args.seed + run
        for K in Ks:
            for algo in algos:
                print(f"[Run {run}] Algo={algo} K={K}")
                out_seg_dir = os.path.join(out_root, "seg", algo, f"K{K}")
                out_ovl_dir = os.path.join(out_root, "overlay", algo, f"K{K}")
                ensure_dir(out_seg_dir); ensure_dir(out_ovl_dir)

                for ip in tqdm(image_paths, desc=f"{algo} K={K}"):
                    t0 = time.time()
                    img = imgs_gray[ip]
                    fe_val = None

                    curve_file = None
                    curve_key = f"{algo}_K{K}_run{run}_" + os.path.splitext(os.path.basename(ip))[0]
                    if args.curves and algo != "otsu":
                        curves_dir = os.path.join(out_root, "curves")
                        ensure_dir(curves_dir)
                        curve_file = os.path.join(curves_dir, curve_key + ".png")

                    if algo == "mfwoa":
                        obj = fuzzy_entropy_objective(hists[ip], K=K, s=2.0)
                        Ts, fits = mfwoa_optimize([obj], [K], pop=max(30, args.pop), iters=args.iters, rmp=args.rmp, seed=run_seed)
                        T = Ts[0]; fe_val = float(fits[0])
                    else:
                        T, best = run_single_algo_on_image(algo, K, img, args.iters, args.pop, run_seed, save_curve=curve_file, curve_key=curve_key)
                        fe_val = float(best) if algo != "otsu" else None

                    seg = apply_thresholds_to_image(img, T, mode='fuzzy', s=2.0)
                    bn = os.path.splitext(os.path.basename(ip))[0]
                    save_gray(os.path.join(out_seg_dir, f"{bn}.png"), (seg * (255 // max(1, seg.max()+1))).astype(np.uint8))
                    try:
                        from PIL import Image
                        ovl = overlay_mask(img, seg, alpha=0.5)
                        Image.fromarray(ovl).save(os.path.join(out_ovl_dir, f"{bn}.png"))
                    except Exception:
                        pass

                    # Metrics
                    mask_path = pairs.get(ip)
                    dsc = iou_val = P = S = None
                    if mask_path is not None and os.path.exists(mask_path):
                        gt = read_gray(mask_path)
                        # Heuristic chuẩn hoá GT đơn giản
                        gt = (gt / max(1, gt.max())).round().astype(np.uint8) if gt.max() > 1 and len(np.unique(gt))<=3 else gt
                        try:
                            perm = hungarian_match(seg, gt)
                            seg_mapped = np.take(perm, seg, mode='clip')
                        except Exception:
                            seg_mapped = seg
                        dsc, _ = dice_score(seg_mapped, gt)
                        iou_val, _ = iou_score(seg_mapped, gt)
                        P = psnr(seg_mapped*(255//(seg_mapped.max()+1)), gt*(255//(gt.max()+1)))
                        try:
                            S = ssim(seg_mapped*(255//(seg_mapped.max()+1)), gt*(255//(gt.max()+1)))
                        except Exception:
                            S = None

                    with open(metrics_path, "a", newline="") as f:
                        wr = csv.writer(f)
                        wr.writerow([ip, algo, K, run, fe_val, dsc, iou_val, P, S, list(map(int, T))])

                    summary_rows.append([ip, algo, K, run, dsc, iou_val, fe_val, time.time()-t0])

    # Summary & charts
    if args.summary and len(summary_rows) > 0:
        import statistics
        from collections import defaultdict
        agg = defaultdict(lambda: {"dice": [], "iou": [], "fe": [], "sec": []})
        per_img = defaultdict(dict)  # per image per algoK → (dice,iou,fe)

        for ip, algo, K, run, dsc, iou_val, fe_val, sec in summary_rows:
            key = f"{algo}_K{K}"
            if dsc is not None: agg[key]["dice"].append(float(dsc))
            if iou_val is not None: agg[key]["iou"].append(float(iou_val))
            if fe_val is not None: agg[key]["fe"].append(float(fe_val))
            agg[key]["sec"].append(float(sec))
            per_img[(ip, K)][algo] = (dsc, iou_val, fe_val)

        sum_path = os.path.join(out_root, f"summary_{ts}.csv")
        with open(sum_path, "w", newline="") as sf:
            sw = csv.writer(sf); sw.writerow(["algo_K","mean_dice","mean_iou","mean_FE","mean_sec","n"])
            for key, v in sorted(agg.items()):
                n = max(len(v["dice"]), len(v["iou"]), len(v["fe"]))
                md = statistics.mean(v["dice"]) if len(v["dice"])>0 else None
                mi = statistics.mean(v["iou"])  if len(v["iou"])>0 else None
                mf = statistics.mean(v["fe"])   if len(v["fe"])>0  else None
                ms = statistics.mean(v["sec"])  if len(v["sec"])>0 else None
                sw.writerow([key, md, mi, mf, ms, n])

        # Charts
        try:
            import matplotlib.pyplot as plt
            # FE bars (no GT)
            labels, fe_vals = [], []
            for key, v in sorted(agg.items()):
                if len(v["fe"])>0:
                    labels.append(key); fe_vals.append(sum(v["fe"])/len(v["fe"]))
            if len(fe_vals)>0:
                plt.figure(); plt.bar(labels, fe_vals); plt.xticks(rotation=45, ha='right')
                plt.ylabel('Mean Fuzzy Entropy'); plt.title('MFWOA vs Baselines (FE)')
                plt.tight_layout(); plt.savefig(os.path.join(out_root, f"summary_FE_{ts}.png")); plt.close()
            # Dice bars (with GT)
            labels, dice_vals = [], []
            for key, v in sorted(agg.items()):
                if len(v["dice"])>0:
                    labels.append(key); dice_vals.append(sum(v["dice"])/len(v["dice"]))
            if len(dice_vals)>0:
                plt.figure(); plt.bar(labels, dice_vals); plt.xticks(rotation=45, ha='right')
                plt.ylabel('Mean Dice'); plt.title('MFWOA vs Baselines (Dice)')
                plt.tight_layout(); plt.savefig(os.path.join(out_root, f"summary_Dice_{ts}.png")); plt.close()
        except Exception as e:
            print("[warn] cannot plot summary charts:", e)

        # Wilcoxon signed-rank: MFWOA vs từng baseline (Dice/IoU)
        if args.sigtest:
            try:
                from scipy.stats import wilcoxon
                sig_path = os.path.join(out_root, f"sigtest_{ts}.csv")
                with open(sig_path, "w", newline="") as sf:
                    sw = csv.writer(sf); sw.writerow(["metric","K","mfwoa_vs","n_pairs","W_stat","p_value"])
                    baselines = [a for a in algos if a != "mfwoa"]
                    for K in Ks:
                        for met_idx, met in enumerate(["dice","iou"]):
                            x_mfwoa, y_base, who = [], [], []
                            for (ip, k), d in per_img.items():
                                if k != K: continue
                                if "mfwoa" in d and any(b in d for b in baselines):
                                    for base in baselines:
                                        if base in d:
                                            mval = d["mfwoa"][met_idx]
                                            bval = d[base][met_idx]
                                            if mval is not None and bval is not None:
                                                x_mfwoa.append(mval); y_base.append(bval); who.append(base)
                                    # chỉ lấy cặp đầu tiên mỗi baseline để không lặp
                            # Test cho từng baseline
                            for base in baselines:
                                xv, yv = [], []
                                for (ipk, d) in per_img.items():
                                    pass
                            # Làm test từng baseline riêng rẽ
                            for base in baselines:
                                X, Y = [], []
                                for (ip, k), d in per_img.items():
                                    if k != K: continue
                                    if "mfwoa" in d and base in d:
                                        mval = d["mfwoa"][met_idx]; bval = d[base][met_idx]
                                        if mval is not None and bval is not None:
                                            X.append(mval); Y.append(bval)
                                if len(X) >= 5:
                                    W, p = wilcoxon(X, Y, zero_method="wilcox", alternative="greater")
                                    sw.writerow([met, K, base, len(X), float(W), float(p)])
            except Exception as e:
                print("[warn] cannot run wilcoxon:", e)
                
        from scipy.stats import wilcoxon
        with open(sig_path, "a", newline="") as sf:
            sw = csv.writer(sf)
            # header đã có; chỉ bổ sung dòng mới cho metric='fe'
            baselines = [a for a in algos if a != "mfwoa"]
            for K in Ks:
                for base in baselines:
                    X, Y = [], []
                    for (ip, k), d in per_img.items():
                        if k != K: 
                            continue
                        if "mfwoa" in d and base in d:
                            mf = d["mfwoa"][2]  # index 2 = FE theo cách mình lưu (dice, iou, fe)
                            bv = d[base][2]
                            if mf is not None and bv is not None:
                                X.append(mf); Y.append(bv)
                    if len(X) >= 5:
                        W, p = wilcoxon(X, Y, zero_method="wilcox", alternative="greater")
                        sw.writerow(["fe", K, base, len(X), float(W), float(p)])

    print(f"Done. Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
