import os
import sys
import subprocess
import numpy as np
import importlib.util
from PIL import Image

# ========== 确保 nltk 可用 ==========
def ensure_nltk():
    """检查并安装 nltk"""
    try:
        import nltk
        print(f"[INFO] 已检测到 nltk (version={nltk.__version__})")
        return nltk
    except ImportError:
        print("[WARN] 未检测到 nltk，正在尝试安装 (使用清华源)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-i",
                                   "https://pypi.tuna.tsinghua.edu.cn/simple", "nltk"])
            import nltk
            print(f"[OK] nltk 安装完成 (version={nltk.__version__})")
            return nltk
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 安装 nltk 失败: {e}")
            sys.exit(1)

# 导入 nltk 和 wordnet
nltk = ensure_nltk()
from nltk.corpus import wordnet as wn

# ========== NLTK 路径配置 ==========
NLTK_PATH = "/opt/data/private/xtt_OPS/OPS/tools/open_miou"
nltk.data.path.append(NLTK_PATH)

# ========== WordNet 工具 ==========
def get_synset(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    return synsets[0] if synsets else None

def build_similarity_matrix(class_names):
    n = len(class_names)
    S = np.zeros((n, n))
    synsets = [get_synset(c) for c in class_names]
    for i in range(n):
        for j in range(n):
            if synsets[i] and synsets[j]:
                sim = synsets[i].path_similarity(synsets[j])
                S[i, j] = sim if sim is not None else 0.0
            else:
                S[i, j] = 0.0
    return S

def compute_iou(pred_mask, gt_mask, class_id):
    pred = (pred_mask == class_id).astype(np.uint8)
    gt = (gt_mask == class_id).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0

def compute_miou(pred_mask, gt_mask, class_names):
    ious = [compute_iou(pred_mask, gt_mask, c) for c in range(len(class_names))]
    return np.mean(ious)

def compute_open_miou(pred_mask, gt_mask, class_names, sim_matrix):
    n = len(class_names)
    open_ious = []
    for g in range(n):
        for p in range(n):
            iou = compute_iou(pred_mask, gt_mask, g) if p == g else 0.0
            sim = sim_matrix[p, g]
            open_ious.append(iou * sim if iou > 0 else 0.0)
    return np.mean(open_ious)

# ========== 从 config 读取配置 ==========
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# ========== 找到最近一次 run_dir ==========
def get_latest_run_dir(work_dir):
    subdirs = [os.path.join(work_dir, d) for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    subdirs = [d for d in subdirs if os.path.basename(d).isdigit() or "_" in os.path.basename(d)]  # 时间戳目录
    if not subdirs:
        raise FileNotFoundError(f"[ERROR] {work_dir} 下没有找到任何时间戳目录")
    return max(subdirs, key=os.path.getmtime)

# ========== 主函数 ==========
def main():
    CONFIG = "/opt/data/private/xtt_OPS/OPS/configs/ops/ops-vit-b16_m3d-640x640.py"

    # 1. 读取 config
    cfg = load_config(CONFIG)
    class_names = cfg.model["text_encoder"]["vocabulary"]
    data_root = cfg.train_dataloader["dataset"]["data_root"]
    seg_map_path = cfg.train_dataloader["dataset"]["data_prefix"]["seg_map_path"]
    gt_dir = os.path.join(data_root, seg_map_path)

    # 2. 找到最近 run_dir 的 preds
    work_root = "/opt/data/private/xtt_OPS/OPS/work_dirs"
    cfg_name = os.path.splitext(os.path.basename(CONFIG))[0]
    exp_dir = os.path.join(work_root, cfg_name)
    run_dir = get_latest_run_dir(exp_dir)
    pred_dir = os.path.join(run_dir, "preds")
    out_file = os.path.join(run_dir, "open_miou_eval.txt")

    print(f"[INFO] 类别数: {len(class_names)}")
    print(f"[INFO] GT 路径: {gt_dir}")
    print(f"[INFO] 最新 run_dir: {run_dir}")
    print(f"[INFO] 预测路径: {pred_dir}")

    # 3. 构建相似度矩阵
    sim_matrix = build_similarity_matrix(class_names)

    # 4. 批量计算
    file_list = sorted(os.listdir(gt_dir))
    miou_list, open_miou_list = [], []

    for fname in file_list:
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)
        if not os.path.exists(pred_path):
            print(f"[WARN] Missing pred for {fname}, skip.")
            continue

        gt_mask = np.array(Image.open(gt_path))
        pred_mask = np.array(Image.open(pred_path))

        miou_list.append(compute_miou(pred_mask, gt_mask, class_names))
        open_miou_list.append(compute_open_miou(pred_mask, gt_mask, class_names, sim_matrix))

    mean_miou = float(np.mean(miou_list)) if miou_list else 0.0
    mean_open_miou = float(np.mean(open_miou_list)) if open_miou_list else 0.0

    # 5. 打印 & 保存结果
    result_str = (
        "========== M3D Open mIoU Evaluation ==========\n"
        f"普通 mIoU:      {mean_miou:.4f}\n"
        f"open mIoU:      {mean_open_miou:.4f}\n"
        "==============================================\n"
    )
    print(result_str)

    with open(out_file, "w") as f:
        f.write(result_str)
    print(f"[INFO] 结果已保存到: {out_file}")

if __name__ == "__main__":
    main()
