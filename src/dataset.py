import os, glob, numpy as np
from pathlib import Path

# Thêm .gif, .tif, .tiff
IMG_EXT = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.gif')

def find_images(root, images_glob=None, debug=False):
    root = str(root)
    if images_glob is None:
        files = []
        for ext in IMG_EXT:
            files.extend(glob.glob(os.path.join(root, f"*{ext}")))
        if len(files) == 0:
            for ext in IMG_EXT:
                files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    else:
        # Hỗ trợ cả backslash & forward-slash trong glob (Windows/Unix)
        pattern1 = os.path.join(root, images_glob)
        pattern2 = os.path.join(root, images_glob.replace('\\', '/'))
        files = glob.glob(pattern1, recursive=True)
        if len(files) == 0 and pattern2 != pattern1:
            files = glob.glob(pattern2, recursive=True)
        # Fallback: quét toàn bộ rồi lọc theo đuôi hợp lệ
        if len(files) == 0:
            files = glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
            files = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXT]

    files = sorted(list(set(files)))
    if debug:
        print(f"[find_images] root={root}")
        print(f"[find_images] images_glob={images_glob}")
        print(f"[find_images] found={len(files)}")
        if len(files) > 0:
            print("  sample:", files[:3])
    return files

def pair_masks(image_paths, root, masks_glob=None):
    pairs = {}
    mask_candidates = []
    if masks_glob is not None:
        mask_candidates = sorted(glob.glob(os.path.join(root, masks_glob), recursive=True))

    # Lập chỉ mục mask theo basename (không extension)
    index = {}
    for m in mask_candidates:
        bn = os.path.splitext(os.path.basename(m))[0]
        index[bn] = m

    # Danh sách phần mở rộng thử cho mask
    TRY_EXTS = ['.png','.jpg','.jpeg','.bmp','.tif','.tiff','.gif']

    for ip in image_paths:
        bn = os.path.splitext(os.path.basename(ip))[0]
        m = index.get(bn)

        if m is None:
            # Chuẩn hoá để replace ổn định (Windows/Unix)
            ip_norm = ip.replace("\\", "/")

            # Thử thay 'images' -> 'masks' hoặc 'mask'
            guess_masks = ip_norm.replace("/images/", "/masks/")
            guess_mask  = ip_norm.replace("/images/", "/mask/")

            candidates = []

            # Nếu đã đổi path, thử các đuôi phổ biến với cùng basename
            for guess in [guess_masks, guess_mask]:
                if guess != ip_norm:
                    d = os.path.dirname(guess)
                    base = os.path.splitext(os.path.basename(guess))[0]
                    for ext in TRY_EXTS:
                        candidates.append(os.path.join(d, base + ext))

            # Nếu ảnh & mask cùng thư mục: thử hậu tố _mask
            dirn = os.path.dirname(ip)
            for ext in TRY_EXTS:
                candidates.append(os.path.join(dirn, bn + "_mask" + ext))

            # Chọn ứng viên đầu tiên tồn tại
            for cand in candidates:
                if os.path.exists(cand):
                    m = cand
                    break

        pairs[ip] = m if (m is not None and os.path.exists(m)) else None
    return pairs

def read_gray(path):
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except Exception:
        pass
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)
