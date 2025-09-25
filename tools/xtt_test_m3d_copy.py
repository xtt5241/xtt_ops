import os
import sys
import subprocess
from datetime import datetime

def main():
    # 固定参数
    CONFIG = "/opt/data/private/xtt_OPS/OPS/configs/ops/ops-vit-b16_m3d-640x640.py"
    # CHECKPOINT = "/opt/data/private/xtt_OPS/OPS/work_dirs/ops-vit-b16_coco-stuff164k-640x640_RERP/iter_59500.pth"
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
    cfg_name = os.path.splitext(os.path.basename(CONFIG))[0]  # ops-vit-b16_m3d-640x640
    work_root = "/opt/data/private/xtt_OPS/OPS/work_dirs"
    exp_dir = os.path.join(work_root, cfg_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(exp_dir, ts)           # e.g. .../ops-vit-b16_m3d-640x640/20250917_232457
    pred_dir = os.path.join(run_dir, "preds")     # e.g. .../20250917_232457/preds
    os.makedirs(pred_dir, exist_ok=True)

    # 构建命令（同时指定 --work-dir 和 --show-dir）
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
        "--work-dir", run_dir,      # 让日志/可视化都进这个时间戳目录
        "--show-dir", pred_dir      # 预测的mask统一存到 run_dir/preds
        # 如果想另存numpy结果，可加： "--out", os.path.join(run_dir, "results.pkl")
    ]

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"\n✅ 预测已保存到: {pred_dir}")
    print(f"   日志与可视化: {run_dir}")

if __name__ == "__main__":
    main()
