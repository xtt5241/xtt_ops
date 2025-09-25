# 项目构成
```
OPS
├── ops
├── configs
├── pretrains
│   ├── ViT-B-16.pt
├── data
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
│   ├── matterport3d
│   │   ├── val
│   │   │   ├── rgb
│   │   │   ├── semantic
│   ├── s2d3d
│   │   ├── area1_3_6
│   │   │   ├── rgb
│   │   │   ├── semantic
│   │   ├── area2_4
│   │   │   ├── rgb
│   │   │   ├── semantic
│   │   ├── area5
│   │   │   ├── rgb
│   │   │   ├── semantic
│   ├── WildPASS
│   │   ├── images
│   │   │   ├── val
│   │   ├── annotations
│   │   │   ├── val
├── tools
├── README.md
```




# train
bash /opt/data/private/xtt_OPS/OPS/tools/dist_train.sh /opt/data/private/xtt_OPS/OPS/configs/ops/ops-vit-b16_coco-stuff164k-640x640.py 4

## 若无法训练，报错，则执行以下命令
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory

# 后续还是有好多问题，这个不要用
# conda activate OPS
# pip uninstall -y opencv-python opencv-contrib-python || true
# pip install --no-cache-dir opencv-python-headless==4.8.1.78"

# 用这个
apt-get update
apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

