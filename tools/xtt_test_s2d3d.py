import os
import sys
import subprocess

def main():
    # 固定参数
    CONFIG = "/opt/data/private/xtt_OPS/OPS/configs/ops/ops-vit-b16_s2d3d-640x640.py"
    CHECKPOINT = "/opt/data/private/xtt_OPS/OPS/work_dirs/ops-vit-b16_coco-stuff164k-640x640_RERP/best_mIoU_iter_59000.pth"
    
    # CHECKPOINT = "/opt/data/private/xtt_OPS/OPS/work_dirs/ops-vit-b16_coco-stuff164k-640x640/iter_60000.pth"
    # CHECKPOINT = "/opt/data/private/xtt_OPS/OPS/OOOPS/OOOPS_without_REPR.pt"
    
    GPUS = "4"

    # 可选环境变量
    NNODES = os.environ.get("NNODES", "1")
    NODE_RANK = os.environ.get("NODE_RANK", "0")
    PORT = os.environ.get("PORT", "29500")
    MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")

    # 设置 PYTHONPATH
    python_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ["PYTHONPATH"] = f"{python_path}:{os.environ.get('PYTHONPATH','')}"

    # 构建命令
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
        "--launcher", "pytorch"
    ]

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
