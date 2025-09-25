# 进入你的项目目录
cd /opt/data/private/xtt_OPS/OPS

# 1) 临时设置国内镜像（当前 shell 有效），并加上 PyTorch cu118 轮子源
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

# 2) 升级 pip，装 PyTorch 2.0.1 + cu118（与你容器 CUDA11.8 匹配）和 torchvision
python3 -m pip install -U pip wheel setuptools
pip install torch==2.0.1 torchvision==0.15.2

# 3) 装你列出的依赖（纯 pip，走阿里云镜像）
pip install lit==18.1.8 numpy==1.23.1 cmake==3.30.4
pip install openmim==0.3.9

# 4) 用 mim 安装 mm 系列指定版本（它内部也是走 pip，会继承上面的镜像设置）
mim install mmengine==0.9.0
mim install mmcv==2.1.0
mim install mmsegmentation==1.2.2

# 5) 其他三方包
pip install timm==0.9.8 einops==0.7.0 ftfy==6.1.1 pkbar==0.5 prettytable==3.9.0 py360convert==0.1.0 regex==2023.10.3 six==1.16.0

# 6) （可选）编译 DCNv3；若失败见下“只在报错时再执行的 2 行”
cd /opt/data/private/xtt_OPS/OPS/ops/models/dcnv3 && bash make.sh





# # 国内镜像 + PyTorch 官方轮子源
# export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

# # 升级 pip 工具
# pip install -U pip setuptools wheel

# # PyTorch 2.0.1 + cu118
# pip install torch==2.0.1 torchvision==0.15.2

# # 其他依赖
# pip install lit==18.1.8 numpy==1.23.1 cmake==3.30.4

# # mm 系列（直接用 pip，不用 mim，避免 SSL 问题）
# # 用阿里云镜像 + PyTorch cu118 轮子源（已装过 torch2.0.1+cu118）

# # 使用阿里云 PyPI + PyTorch cu118 源
# export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

# # 从 OpenMMLab 官方轮子索引安装（跳过证书校验）
# pip install "mmcv==2.1.0" \
#   --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
#   --trusted-host download.openmmlab.com


# # 安装（关键：用 /usr/bin/python -m pip，确保装进同一个解释器）
# /usr/bin/python -m pip install -U pip setuptools wheel
# /usr/bin/python -m pip install mmsegmentation==1.2.2


# # 额外第三方包
# pip install timm==0.9.8 einops==0.7.0 ftfy==6.1.1 pkbar==0.5 \
#             prettytable==3.9.0 py360convert==0.1.0 regex==2023.10.3 six==1.16.0

