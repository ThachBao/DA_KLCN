# Phân đoạn ảnh xám theo Fuzzy Entropy (MFWOA vs Otsu/PSO/GA/WOA)

## Mục tiêu
1) Cài đặt & thực nghiệm phân đoạn đa ngưỡng với Fuzzy Entropy, đánh giá **Dice, IoU, PSNR, SSIM** (nếu có GT) và **overlay** trực quan.
2) So sánh **MFWOA** với **Otsu/PSO/GA/WOA**, tổng hợp & kiểm định **ý nghĩa thống kê** để chứng minh tính ưu việt.

## Cài đặt
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt


Dữ liệu mẫu

```
DATASET_ROOT/
  images/*.jpg|png|bmp|tif
  mask(s)/*.png             # tên trùng ảnh (nếu có GT)
```


3) Chạy



```

Example when you **do** have masks (paired names):
```bash
python -m src.cli.run_experiment `
  --dataset_root "C:\Zalo Received Files\KLCN\DA_KLCN_fixed\dataset" `
  --images_glob "images/**/*.*" `
  --masks_glob  "mask/**/*.*" `
  --out "results" `
  --algos mfwoa,woa,pso,ga,otsu `
  --Ks 2,3 `
  --iters 200 --pop 50 --runs 3 --seed 42 `
  --summary --curves --sigtest --debug_glob


```

4) Results
Mask: results/seg/{algo}/K{k}/*.png

Overlay: results/overlay/{algo}/K{k}/*.png

Chỉ số từng ảnh: results/metrics_*.csv

Tổng hợp: results/summary_*.csv, summary_FE_*.png, summary_Dice_*.png

Thống kê: results/sigtest_*.csv

Report Markdown:

---

## Requirements

python -m src.cli.make_report --in "results" --out "results\report.md"

