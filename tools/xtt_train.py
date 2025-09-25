import os
import sys
import subprocess

# --------------------------
# 配置参数
# --------------------------
# CONFIG = "/opt/data/private/xtt_OPS/OPS/configs/ops/ops-vit-b16_coco-stuff164k-640x640_RERP.py"
# CONFIG = "/opt/data/private/xtt_OPS/OPS/configs/ops/ops-vit-b16_coco-stuff164k-640x640_ERP.py"
CONFIG = "/opt/data/private/xtt_OPS/OPS/configs/ops/ops-vit-b16_coco-stuff164k-640x640_RERP_ca.py"
GPUS = 4

NNODES = int(os.environ.get("NNODES", 1))
NODE_RANK = int(os.environ.get("NODE_RANK", 0))
PORT = int(os.environ.get("PORT", 29500))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
train_script = os.path.join(script_dir, "train.py")

# PYTHONPATH 设置
pythonpath = os.path.join(script_dir, "..")
os.environ["PYTHONPATH"] = f"{pythonpath}:{os.environ.get('PYTHONPATH','')}"

# 传递给 train.py 的额外参数
extra_args = sys.argv[1:]

# 构建分布式训练命令
cmd = [
    "python", "-m", "torch.distributed.launch",
    f"--nnodes={NNODES}",
    f"--node_rank={NODE_RANK}",
    f"--master_addr={MASTER_ADDR}",
    f"--nproc_per_node={GPUS}",
    f"--master_port={PORT}",
    train_script,
    CONFIG,
    "--launcher", "pytorch"
] + extra_args

# 打印命令，方便调试
print("Running command:")
print(" ".join(cmd))

# 执行命令
subprocess.run(cmd)
