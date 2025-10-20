# Phân đoạn ảnh xám đa ngưỡng bằng Fuzzy Entropy (MFWOA vs Otsu/PSO/GA/WOA)

> **Mục tiêu**: Cài đặt – thực nghiệm phân đoạn đa ngưỡng dựa trên **Fuzzy Entropy**; so sánh **MFWOA** với **Otsu/PSO/GA/WOA** qua các chỉ số định lượng và trực quan hóa kết quả.

---

## 🔑 Tính năng chính
- **Thuật toán**: MFWOA (Modified Fuzzy Whale Optimization) và các đối chứng Otsu/PSO/GA/WOA.
- **Đánh giá**: Dice, IoU, PSNR, SSIM *(khi có ground truth)*.
- **Trực quan**: Xuất **mask** và **overlay** lên ảnh gốc, biểu đồ tổng hợp.
- **Thống kê**: Kiểm định ý nghĩa thống kê giữa các thuật toán.
- **Báo cáo**: Tự động tổng hợp báo cáo Markdown từ thư mục kết quả.

---

## 🧰 Chuẩn bị môi trường
### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Gợi ý**: Nếu dùng Unix/macOS, kích hoạt môi trường bằng `source .venv/bin/activate`.

---

## 🗂️ Dữ liệu đầu vào
Tối thiểu cần một thư mục ảnh và **tuỳ chọn** có ground truth (mask) trùng tên để tính các chỉ số:

```
DATASET_ROOT/
  images/*.jpg|png|bmp|tif
  mask/*.png                # nếu có GT, tên trùng ảnh
```

---

## ▶️ Chạy thực nghiệm
Ví dụ khi **có** ground truth (mask) với cấu trúc như trên:

```bash
python -m src.cli.run_experiment --dataset_root "C:\Zalo Received Files\KLCN_ver1\DA_KLCN _ver1\dataset" --images_glob 'images\**\*.*' --masks_glob 'mask\**\*.*' --out "results" --algos mfwoa,woa,pso,ga,otsu --Ks 2,3 --iters 200 --pop 50 --runs 3 --seed 42 --summary --curves --sigtest --debug_glob
```

### Tham số quan trọng
- `--algos`: danh sách thuật toán cần so sánh.
- `--Ks`: số mức ngưỡng (ví dụ `2,3`).
- `--iters`, `--pop`: số vòng lặp và kích thước quần thể cho metaheuristics.
- `--runs`: số lần chạy lặp lại để lấy trung bình/độ lệch chuẩn.
- `--seed`: cố định hạt giống ngẫu nhiên để tái lập.
- Cờ tiện ích:
  - `--summary`: xuất bảng tổng hợp.
  - `--curves`: vẽ biểu đồ tổng hợp.
  - `--sigtest`: kiểm định ý nghĩa thống kê.
  - `--debug_glob`: log chi tiết lọc ảnh theo glob.

> **Lưu ý**: Nếu đường dẫn chứa khoảng trắng, hãy **đặt trong dấu nháy** như ví dụ trên.

---

## 📦 Kết quả đầu ra
Sau khi chạy, các kết quả sẽ được sắp xếp trong thư mục `results/`:

- **Mask phân đoạn**: `results/seg/{algo}/K{k}/*.png`
- **Overlay (mask đè lên ảnh gốc)**: `results/overlay/{algo}/K{k}/*.png`
- **Chỉ số từng ảnh**: `results/metrics_*.csv`
- **Tổng hợp**: `results/summary_*.csv`, `results/summary_FE_*.png`, `results/summary_Dice_*.png`
- **Thống kê**: `results/sigtest_*.csv`

---

## 📝 Tạo báo cáo Markdown
Sinh báo cáo tổng hợp (Markdown) từ thư mục `results/`:

```bash
python -m src.cli.make_report --in "results" --out "results/report.md"
```

---

## 🗃️ Gợi ý cấu trúc dự án
*(tham khảo – có thể khác tuỳ repo)*
```
├─ src/
│  ├─ algo/                 # MFWOA, WOA, PSO, GA, Otsu
│  ├─ metrics/              # Dice, IoU, PSNR, SSIM
│  ├─ viz/                  # overlay, biểu đồ, tổng hợp
│  └─ cli/                  # run_experiment, make_report
├─ dataset/                 # images/, mask/
├─ results/                 # seg/, overlay/, metrics_*.csv, ...
├─ requirements.txt
└─ README.md
```

---

## ❓ FAQ / Troubleshooting
- **Không đọc được ảnh**: kiểm tra `--images_glob` có khớp phần mở rộng (jpg/png/bmp/tif) và thư mục.
- **Không có mask/không khớp tên**: cần đảm bảo tên file mask trùng tên ảnh; nếu không có GT, một số chỉ số (Dice/IoU) sẽ không tính được.
- **Kết quả khác nhau giữa các lần chạy**: tăng `--runs` hoặc cố định `--seed` để ổn định.
- **Thiếu quyền ghi**: kiểm tra quyền ghi vào thư mục `--out`.

---

## 📄 Giấy phép & Trích dẫn
Nếu sử dụng mã nguồn/kết quả trong bài báo hoặc đồ án, hãy trích dẫn phù hợp theo chuẩn bạn sử dụng (APA/IEEE/ACM). Thêm thông tin giấy phép vào đây (MIT/BSD/GPL…).

---

## 💬 Liên hệ & Đóng góp
Vui lòng tạo **issue** hoặc **pull request** nếu bạn phát hiện lỗi, cần tính năng mới, hoặc muốn đóng góp cải tiến.

