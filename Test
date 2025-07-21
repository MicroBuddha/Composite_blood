import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Increase default DPI for matplotlib figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ====================================
# 1) CONFIGURATION: PATHS & CONSTANTS
# ====================================
BASE_PATH   = "/home/buddhadev/Desktop/afnan_blood/combined_dataset_v2/test"
BG_DIR      = os.path.join(BASE_PATH, "background_images")
SEGMENT_DIR = os.path.join(BASE_PATH, "segments")
RBC_DIR     = os.path.join(BASE_PATH, "rbc")
MODEL_PATH  = os.path.join(BASE_PATH, "/home/buddhadev/Desktop/afnan_blood/best_yolo_grayscale.pt")
N_IMAGES    = 500

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet", "lymphoblast"
]
CLASS_NAME_CORRECTIONS = {
    "immunoglobin":         "ig",
    "immature-granulocyte": "ig"
}

SEGMENTS_PER_IMAGE_RANGE = (15, 35)
RBC_PER_IMAGE_RANGE      = (30, 80)

# Ensure at least one background
if not os.path.exists(BG_DIR) or not glob.glob(os.path.join(BG_DIR, '*')):
    os.makedirs(BG_DIR, exist_ok=True)
    dummy = np.ones((1000,1000,3), dtype=np.uint8) * 240
    cv2.imwrite(os.path.join(BG_DIR, 'dummy_bg.jpg'), dummy)

# ====================================
# 2) IMAGE TRANSFORM HELPERS
# ====================================
def to_grayscale_with_alpha(img):
    if img.shape[2] == 4:
        bgr = img[..., :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        alpha = img[..., 3]
        return np.dstack((gray3, alpha))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_noise(image, gauss_sigma=30, smooth_ksize=(1,1), poisson_strength=0.1):
    if image.shape[2] == 4:
        alpha = image[:, :, 3].copy()
        img = image[:, :, :3].astype(np.float32)
    else:
        img = image.astype(np.float32)
        alpha = None
    gauss = np.random.normal(0, gauss_sigma, img.shape).astype(np.float32)
    gauss = cv2.GaussianBlur(gauss, smooth_ksize, 0)
    noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)
    if poisson_strength > 0:
        mask = np.random.rand(*noisy.shape[:2]) < poisson_strength
        norm = noisy.astype(np.float32) / 255.0
        shot = np.random.poisson(norm * 255.0) / 255.0
        shot = np.clip(shot * 255.0, 0, 255).astype(np.uint8)
        noisy[mask] = shot[mask]
    if alpha is not None:
        out = cv2.cvtColor(noisy, cv2.COLOR_BGR2BGRA)
        out[:, :, 3] = alpha
        return out
    return noisy


def rotate_image(img, angle=None):
    h, w = img.shape[:2]
    a = angle if angle is not None else random.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), a, 1.0)
    border = (0, 0, 0, 0) if img.shape[2] == 4 else (0, 0, 0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)

# Transform names
TEST_PARAMETERS = {
    "original":      "Original",
    "grayscale":     "Grayscale",
    "blur_light":    "Gaussian Blur (5×5)",
    "blur_heavy":    "Gaussian Blur (15×15)",
    "brightness_up": "+Brightness",
    "brightness_down":"-Brightness",
    "contrast_up":   "+Contrast",
    "contrast_down": "-Contrast",
    "noise":         "Gaussian(σ=30)+Poisson(10%)",
    "scale_down":    "Scale ×0.8",
    "scale_up":      "Scale ×1.2",
    "scale_down_05": "Scale ×0.5",
    "scale_up_2":    "Scale ×2.0",
    "rotation":      "Rotate ±45°"
}

# ====================================
# 3) UTILS: Overlap & IoU
# ====================================
def is_overlapping(box, boxes, padding=0):
    x1, y1, x2, y2 = box
    x1 -= padding; y1 -= padding; x2 += padding; y2 += padding
    for bx1, by1, bx2, by2 in boxes:
        if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
            return True
    return False


def calculate_iou(b1, b2):
    xL = max(b1[0], b2[0]); yT = max(b1[1], b2[1])
    xR = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    if xR < xL or yB < yT:
        return 0.0
    inter = (xR - xL) * (yB - yT)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0

# ====================================
# 4) EVALUATION: Detection Metrics
# ====================================
def evaluate_detection(gt_cells, yolo_res, iou_threshold=0.5):
    class_metrics = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CLASS_NAMES}
    y_true, y_pred = [], []
    if yolo_res is None or len(yolo_res) == 0:
        for gt in gt_cells:
            class_metrics[gt['class']]['fn'] += 1
            y_true.append(gt['class']); y_pred.append('missed')
    else:
        res0 = yolo_res[0]
        matched = [False] * len(gt_cells)
        for box in getattr(res0, 'boxes', []):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cid = int(box.cls[0]); pred = res0.names[cid]
            pred = CLASS_NAME_CORRECTIONS.get(pred, pred)
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gt_cells):
                if matched[i]:
                    continue
                iou = calculate_iou([x1, y1, x2, y2], gt['box'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou, best_idx = iou, i
            if best_idx >= 0:
                true_cls = gt_cells[best_idx]['class']; matched[best_idx] = True
                if pred == CLASS_NAME_CORRECTIONS.get(true_cls, true_cls):
                    class_metrics[true_cls]['tp'] += 1
                else:
                    class_metrics[true_cls]['fn'] += 1
                    class_metrics[pred]['fp'] += 1 if pred in class_metrics else 0
                y_true.append(true_cls); y_pred.append(pred)
            else:
                class_metrics[pred]['fp'] += 1
                y_true.append('background'); y_pred.append(pred)
        for i, flag in enumerate(matched):
            if not flag:
                cls = gt_cells[i]['class']
                class_metrics[cls]['fn'] += 1
                y_true.append(cls); y_pred.append('missed')
    tp = sum(v['tp'] for v in class_metrics.values())
    fp = sum(v['fp'] for v in class_metrics.values())
    fn = sum(v['fn'] for v in class_metrics.values())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    cm_data = None
    if y_true:
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
        cm_data = (cm, CLASS_NAMES)
    return {
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'class_metrics': class_metrics,
        'confusion_matrix': cm_data
    }

# ====================================
# 5) UTILITY: get y_true/y_pred pairs
# ====================================
def get_ytrue_ypred(gt_cells, yolo_res, iou_threshold=0.5):
    y_true, y_pred = [], []
    if yolo_res is None or len(yolo_res) == 0:
        for gt in gt_cells:
            y_true.append(gt['class']); y_pred.append('missed')
        return y_true, y_pred
    res0 = yolo_res[0]
    matched = [False] * len(gt_cells)
    for box in getattr(res0, 'boxes', []):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cid = int(box.cls[0]); pred = res0.names[cid]
        pred = CLASS_NAME_CORRECTIONS.get(pred, pred)
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_cells):
            if matched[i]:
                continue
            iou = calculate_iou([x1, y1, x2, y2], gt['box'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou, best_idx = iou, i
        if best_idx >= 0:
            matched[best_idx] = True
            y_true.append(gt_cells[best_idx]['class'])
            y_pred.append(pred)
        else:
            y_true.append('background'); y_pred.append(pred)
    for i, flag in enumerate(matched):
        if not flag:
            y_true.append(gt_cells[i]['class'])
            y_pred.append('missed')
    return y_true, y_pred

# ====================================
# 6) VISUALIZATIONS (with labels)
# ====================================
def visualize_with_both_annotations(img, gt_cells, res, title, save_path):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(rgb)
    ax.set_title(title)
    # Draw GT
    for cell in gt_cells:
        x1, y1, x2, y2 = cell['box']
        rect = Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='lime', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-6, f"GT: {cell['class']}", color='lime', fontsize=8, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))
    # Draw predictions
    if res and hasattr(res[0], 'boxes'):
        res0 = res[0]
        for box in res0.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cid = int(box.cls[0]); cls = res0.names[cid]
            cls = CLASS_NAME_CORRECTIONS.get(cls, cls)
            conf = float(box.conf[0])
            rect = Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y2+6, f"Pred: {cls} ({conf:.2f})", color='red', fontsize=8, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))
    ax.axis('off'); plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

# ====================================
# 7) COMPOSITE GENERATION & PLACEMENT
# ====================================
def place_segments_on_background(bg, segments, rbc_list,
                                 num_cells_per_class, transform_key,
                                 placement_seed, transform_seed):
    # 1) seed randomness
    random.seed(placement_seed)
    np.random.seed(placement_seed)
    canvas = bg.copy()
    h, w = canvas.shape[:2]
    placed = []
    gt_cells = []
    # determine per-cell scale
    scale_map = {
        'scale_down': 0.8,
        'scale_up': 1.2,
        'scale_down_05': 0.5,
        'scale_up_2': 2.0
    }
    scale_f = scale_map.get(transform_key)
    # determine rotation angle once per image
    angle = None
    if transform_key == 'rotation':
        random.seed(transform_seed)
        angle = random.uniform(-45, 45)
    # place each class
    for cls, pool in segments.items():
        if not pool:
            continue
        # sample a few patches
        sel = random.sample(pool, min(num_cells_per_class, len(pool)))
        for seg_img, fp in sel:
            patch = seg_img.copy()
            # apply scale
            if scale_f:
                if patch.ndim < 3:
                    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGRA)
                if patch.shape[2] == 4:
                    alpha_ch = patch[:, :, 3].astype(np.float32) / 255.0
                    bgr = patch[:, :, :3]
                else:
                    bgr = patch
                    alpha_ch = np.ones(bgr.shape[:2], dtype=np.float32)
                sh, sw = bgr.shape[:2]
                nw, nh = int(sw * scale_f), int(sh * scale_f)
                if nw > 0 and nh > 0:
                    bgr = cv2.resize(bgr, (nw, nh))
                    alpha_ch = cv2.resize(alpha_ch, (nw, nh))
                patch = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                patch[:, :, 3] = (alpha_ch * 255).astype(np.uint8)
            # apply rotation
            elif transform_key == 'rotation' and angle is not None:
                patch = rotate_image(patch, angle)
            # apply other transforms
            elif transform_key in ['grayscale','blur_light','blur_heavy',
                                   'brightness_up','brightness_down',
                                   'contrast_up','contrast_down','noise']:
                func_map = {
                    'grayscale': to_grayscale_with_alpha,
                    'blur_light': lambda im: cv2.GaussianBlur(im, (5,5), 0),
                    'blur_heavy': lambda im: cv2.GaussianBlur(im, (15,15), 0),
                    'brightness_up': lambda im: cv2.convertScaleAbs(im, alpha=1.3, beta=30),
                    'brightness_down': lambda im: cv2.convertScaleAbs(im, alpha=0.7, beta=-30),
                    'contrast_up': lambda im: cv2.convertScaleAbs(im, alpha=1.5, beta=0),
                    'contrast_down': lambda im: cv2.convertScaleAbs(im, alpha=0.7, beta=0),
                    'noise': add_noise
                }
                patch = func_map[transform_key](patch)
            # blend patch
            ph, pw = patch.shape[:2]
            if pw == 0 or ph == 0:
                continue
            alpha_ch = patch[:, :, 3].astype(np.float32)/255.0 if patch.shape[2]==4 else np.ones((ph,pw))
            bgr = patch[:, :, :3] if patch.shape[2]==4 else patch
            # try placement
            for _ in range(50):
                x = random.randint(0, w-pw)
                y = random.randint(0, h-ph)
                box = (x, y, x+pw, y+ph)
                if not is_overlapping(box, placed):
                    roi = canvas[y:y+ph, x:x+pw].astype(np.float32)
                    for c in range(3):
                        roi[:,:,c] = roi[:,:,c]*(1-alpha_ch) + bgr[:,:,c]*alpha_ch
                    canvas[y:y+ph, x:x+pw] = roi.astype(np.uint8)
                    placed.append(box)
                    gt_cells.append({'class': cls, 'box': [x, y, x+pw, y+ph]})
                    break
    # place unlabeled RBCs
    random.shuffle(rbc_list)
    for rbc_img, fp in rbc_list[:sum(SEGMENTS_PER_IMAGE_RANGE)]:
        patch = rbc_img.copy()
        ph, pw = patch.shape[:2]
        alpha_ch = patch[:,:,3].astype(np.float32)/255.0 if patch.ndim==3 and patch.shape[2]==4 else 1
        bgr = patch[:,:,:3] if patch.ndim==3 and patch.shape[2]==4 else patch
        for _ in range(50):
            x = random.randint(0, w-pw)
            y = random.randint(0, h-ph)
            box = (x, y, x+pw, y+ph)
            if not is_overlapping(box, placed):
                roi = canvas[y:y+ph, x:x+pw].astype(np.float32)
                for c in range(3):
                    roi[:,:,c] = roi[:,:,c]*(1-alpha_ch) + bgr[:,:,c]*alpha_ch
                canvas[y:y+ph, x:x+pw] = roi.astype(np.uint8)
                placed.append(box)
                break
    return canvas, gt_cells

# ====================================
# 7) MAIN TEST LOOP: Runs, Metrics, Overlays
# ====================================
def run_fullset_tests():
    seeds = [100, 200, 300, 400, 500]
    for run_id, seed in enumerate(seeds, start=1):
        random.seed(seed)
        np.random.seed(seed)
        model = YOLO(MODEL_PATH)
        root = os.path.join(BASE_PATH, f"run{run_id}")
        subdirs = ['transformed', 'samples', 'metrics_img', 'metrics_cls', 'cms']
        for sd in subdirs:
            os.makedirs(os.path.join(root, sd), exist_ok=True)

        # load segments and RBC patches
        segments = {cls: [] for cls in CLASS_NAMES}
        for fp in glob.glob(os.path.join(SEGMENT_DIR, '*')):
            fn = os.path.basename(fp).lower()
            for cls in CLASS_NAMES:
                if fn.startswith(cls):
                    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        segments[cls].append((img, fp))
        rbc_list = [(cv2.imread(fp, cv2.IMREAD_UNCHANGED), fp)
                    for fp in glob.glob(os.path.join(RBC_DIR, '*'))]

        # process each transform
        for tkey, tname in TEST_PARAMETERS.items():
            per_image = []
            agg_counts = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in CLASS_NAMES}
            yts, yps = [], []

            for idx in range(N_IMAGES):
                bg = cv2.imread(random.choice(glob.glob(os.path.join(BG_DIR, '*'))))
                comp, gt_cells = place_segments_on_background(
                    bg, segments, rbc_list,
                    num_cells_per_class=3,
                    transform_key=tkey,
                    placement_seed=seed + idx,
                    transform_seed=seed + idx
                )
                # save composite
                fname = f"{tkey}_{idx:04d}.jpg"
                outp = os.path.join(root, 'transformed', fname)
                cv2.imwrite(outp, comp)

                # inference
                try:
                    res = model(outp)
                except:
                    res = None

                # evaluate
                metr = evaluate_detection(gt_cells, res)
                per_image.append({
                    'transform': tkey,
                    'image':     fname,
                    'precision': metr['precision'],
                    'recall':    metr['recall'],
                    'f1_score':  metr['f1_score']
                })
                # accumulate per-class
                for cls, cnt in metr['class_metrics'].items():
                    agg_counts[cls]['tp'] += cnt['tp']
                    agg_counts[cls]['fp'] += cnt['fp']
                    agg_counts[cls]['fn'] += cnt['fn']
                # accumulate for confusion matrix
                yt, yp = get_ytrue_ypred(gt_cells, res)
                yts.extend(yt)
                yps.extend(yp)

                # sample overlays
                if idx < 5:
                    save_name = f"{tkey}_sample{idx:02d}.png"
                    save_path = os.path.join(root, 'samples', save_name)
                    visualize_with_both_annotations(comp, gt_cells, res,
                                                    f"Run {run_id} - {tname}", save_path)

            # save per-image metrics
            df_img = pd.DataFrame(per_image)
            df_img.to_csv(
                os.path.join(root, 'metrics_img', f"{tkey}_metrics.csv"), index=False)

            # save per-class aggregated metrics
            rows_cls = []
            for cls, cnt in agg_counts.items():
                tp, fp, fn = cnt['tp'], cnt['fp'], cnt['fn']
                prec = tp/(tp+fp) if tp+fp>0 else 0
                rec  = tp/(tp+fn) if tp+fn>0 else 0
                f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
                rows_cls.append({
                    'transform': tkey,
                    'class':     cls,
                    'tp':        tp,
                    'fp':        fp,
                    'fn':        fn,
                    'precision': prec,
                    'recall':    rec,
                    'f1_score':  f1
                })
            pd.DataFrame(rows_cls).to_csv(
                os.path.join(root, 'metrics_cls', f"{tkey}_classwise.csv"), index=False)

            # save confusion matrix
            if yts:
                cm_arr = confusion_matrix(yts, yps, labels=CLASS_NAMES)
                plt.figure(figsize=(10,8))
                sns.heatmap(cm_arr, annot=True, fmt='d',
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix: {tname} Run {run_id}')
                cm_path = os.path.join(root, 'cms', f"{tkey}_cm.png")
                plt.tight_layout()
                plt.savefig(cm_path)
                plt.close()

        print(f"Completed run {run_id}")

if __name__ == '__main__':
    run_fullset_tests()
