# -*- coding: utf-8 -*-
import os
import cv2
from argparse import ArgumentParser
from multiprocessing import Pool
from tqdm import tqdm
from mmseg.registry import TRANSFORMS

# 触发注册（若不在同目录，改成 from tools.dataset_converters.pano_trans import PanoTrans）
from xtt_pano_trans import PanoTrans  # noqa: F401


def parse_args():
    parser = ArgumentParser(description='Add ERP/RERP to coco-stuff164k')
    parser.add_argument('--mode', default='train',
                        help='train or val (train/val). default: train')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers. default: 4')
    parser.add_argument('--shuffle', action='store_true',
                        help='RERP (shuffle) if set; otherwise ERP.')
    return parser.parse_args()


def build_pairs(img_dir, ann_dir):
    # 只收真正的图像文件（避免把 *.png 掩码误当图）
    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg'))]
    ann_files = [f for f in os.listdir(ann_dir)
                 if f.endswith('_labelTrainIds.png')]

    # 基名到全路径
    img_map = {os.path.splitext(f)[0]: os.path.join(img_dir, f) for f in img_files}
    ann_map = {f.replace('_labelTrainIds.png', ''): os.path.join(ann_dir, f)
               for f in ann_files}

    common = sorted(set(img_map.keys()) & set(ann_map.keys()))
    img_paths = [img_map[k] for k in common]
    ann_paths = [ann_map[k] for k in common]

    print(f'paired samples: {len(common)}; '
          f'orphan images: {len(img_map)-len(common)}; '
          f'orphan labels: {len(ann_map)-len(common)}')
    return img_paths, ann_paths


def process_image(i):
    img_path, ann_path = img_paths[i], ann_paths[i]

    # 名称强一致校验（第一时间暴露错位）
    assert os.path.splitext(os.path.basename(img_path))[0] == \
           os.path.basename(ann_path).replace('_labelTrainIds.png', ''), \
        f'name mismatch: {img_path} <-> {ann_path}'

    results = dict(img_path=img_path, seg_map_path=ann_path,
                   reduce_zero_label=False, seg_fields=[])

    # 逐步走管线
    res = load_img(results)
    res = load_ann(res)
    res = resize(res)
    res = crop(res)

    # ERP/RERP
    res = pano(res)

    img = res['img']
    ann = res['gt_seg_map']

    # 尺寸一致性校验
    assert img.shape[:2] == ann.shape[:2], \
        f'size mismatch: {img.shape} vs {ann.shape} @ {img_path}'

    if args.shuffle:
        cv2.imwrite(os.path.join(img_dir_out, 'erp_shuffle_' + os.path.basename(img_path)), img)
        cv2.imwrite(os.path.join(ann_dir_out, 'erp_shuffle_' + os.path.basename(ann_path)), ann)
    else:
        cv2.imwrite(os.path.join(img_dir_out, 'erp_no_shuffle_' + os.path.basename(img_path)), img)
        cv2.imwrite(os.path.join(ann_dir_out, 'erp_no_shuffle_' + os.path.basename(ann_path)), ann)


if __name__ == '__main__':
    args = parse_args()

    # 分辨率按照 (H, W)
    resolution = (640, 640)

    img_dir = f'./data/coco_stuff164k/images/{args.mode}2017'
    ann_dir = f'./data/coco_stuff164k/annotations/{args.mode}2017'

    # —— 用“按基名配对”的方式构造样本对
    img_paths, ann_paths = build_pairs(img_dir, ann_dir)

    # 经典 MMSeg 读入与几何增强
    load_img = TRANSFORMS.build(dict(type='LoadImageFromFile'))
    load_ann = TRANSFORMS.build(dict(type='LoadAnnotations'))
    resize   = TRANSFORMS.build(dict(type='ResizeShortestEdge', scale=resolution, max_size=10_000_000))
    crop     = TRANSFORMS.build(dict(type='RandomCrop', crop_size=resolution, cat_max_ratio=1.0))

    # 输出目录
    if args.shuffle:
        img_dir_out = img_dir + '_erp_shuffle'
        ann_dir_out = ann_dir + '_erp_shuffle'
        os.makedirs(img_dir_out, exist_ok=True)
        os.makedirs(ann_dir_out, exist_ok=True)

        # RERP：先打乱再 ERP
        pano = TRANSFORMS.build(dict(
            type='PanoTrans',
            mode='rerp',
            shuffle=True,
            p_shuffle=0.5,          # 训练可调 0.5~1.0
            grid=2,                 # 或 grid_choices=[2,3]
            fov_deg=90.0,
            strength=1.0,
            out_size=resolution,
            ignore_index=255
        ))
    else:
        img_dir_out = img_dir + '_erp_no_shuffle'
        ann_dir_out = ann_dir + '_erp_no_shuffle'
        os.makedirs(img_dir_out, exist_ok=True)
        os.makedirs(ann_dir_out, exist_ok=True)

        # ERP：不打乱
        pano = TRANSFORMS.build(dict(
            type='PanoTrans',
            mode='erp',
            shuffle=False,
            fov_deg=90.0,
            strength=1.0,
            out_size=resolution,
            ignore_index=255
        ))

    # 跑
    with Pool(processes=args.num_workers) as pool:
        _ = list(tqdm(pool.imap(process_image, range(len(img_paths))),
                      total=len(img_paths)))
