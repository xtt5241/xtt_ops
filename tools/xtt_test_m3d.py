import os
import sys
import subprocess
import numpy as np
import importlib.util
from PIL import Image
from datetime import datetime

# ========== 确保 nltk 可用 ==========
def ensure_nltk():
    try:
        import nltk
        print(f"[INFO] 已检测到 nltk (version={nltk.__version__})")
        return nltk
    except ImportError:
        print("[WARN] 未检测到 nltk，正在尝试安装 (使用清华源)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-i",
                               "https://pypi.tuna.tsinghua.edu.cn/simple", "nltk"])
        import nltk
        print(f"[OK] nltk 安装完成 (version={nltk.__version__})")
        return nltk

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

# ========== 主函数 ==========
def main():
    # 固定参数
    CONFIG = "/opt/data/private/xtt_OPS/OPS/configs/ops/ops-vit-b16_m3d-640x640.py"
    # CHECKPOINT = "/opt/data/private/xtt_OPS/OPS/work_dirs/ops-vit-b16_coco-stuff164k-640x640/iter_60000.pth"
    CHECKPOINT = "/opt/data/private/xtt_OPS/OPS/work_dirs/ops-vit-b16_coco-stuff164k-640x640_ERP/iter_55000.pth"

    GPUS = "4"

    # 可选环境变量
    NNODES = os.environ.get("NNODES", "1")
    NODE_RANK = os.environ.get("NODE_RANK", "0")
    PORT = os.environ.get("PORT", "29500")
    MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")

    # 设置 PYTHONPATH
    python_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ["PYTHONPATH"] = f"{python_path}:{os.environ.get('PYTHONPATH','')}"

    # ===== 关键：构造与 mmengine 一致的时间戳目录 =====
    cfg_name = os.path.splitext(os.path.basename(CONFIG))[0]
    work_root = "/opt/data/private/xtt_OPS/OPS/work_dirs"
    exp_dir = os.path.join(work_root, cfg_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(exp_dir, ts)
    pred_dir = os.path.join(run_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    # ===== 第一步：先跑预测并保存 mask =====
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nnodes={NNODES}",
        f"--node_rank={NODE_RANK}",
        f"--master_addr={MASTER_ADDR}",
        f"--nproc_per_node={GPUS}",
        f"--master_port={PORT}",
        os.path.join(os.path.dirname(__file__), "test.py"),
        CONFIG,
        CHECKPOINT,
        "--launcher", "pytorch",
        "--work-dir", run_dir,
        "--show-dir", pred_dir
    ]

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\n✅ 推理完成，预测已保存到: {pred_dir}")

    # ===== 第二步：跑 open mIoU 评估 =====
    cfg = load_config(CONFIG)
    class_names = cfg.model["text_encoder"]["vocabulary"]
    data_root = cfg.train_dataloader["dataset"]["data_root"]
    seg_map_path = cfg.train_dataloader["dataset"]["data_prefix"]["seg_map_path"]
    gt_dir = os.path.join(data_root, seg_map_path)

    print(f"[INFO] 类别数: {len(class_names)}")
    print(f"[INFO] GT 路径: {gt_dir}")

    sim_matrix = build_similarity_matrix(class_names)

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

    result_str = (
        "========== M3D Open mIoU Evaluation ==========\n"
        f"普通 mIoU:      {mean_miou:.4f}\n"
        f"open mIoU:      {mean_open_miou:.4f}\n"
        "==============================================\n"
    )
    print(result_str)

    out_file = os.path.join(run_dir, "open_miou_eval.txt")
    with open(out_file, "w") as f:
        f.write(result_str)
    print(f"[INFO] 结果已保存到: {out_file}")

if __name__ == "__main__":
    main()
