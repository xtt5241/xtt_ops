# xtt 9.16 add 跳过已经处理的文件

import cv2
import os
from mmseg.registry import TRANSFORMS
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser

# xtt 9.16 add
import numpy as np
import py360convert
import random as rd
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PanoTrans(BaseTransform):

    def __init__(self, ignore_index=255, shuffle=False, crop_size=(640, 640)):
        self.ignore_index = ignore_index
        self.shuffle = shuffle
        self.crop_size = crop_size

    @staticmethod
    def _shuffle(results, key, order):
        img_array = results[key]
        split_height, split_width = img_array.shape[0] // 2, img_array.shape[1] // 2
        img_blocks = [
            img_array[:split_height, :split_width],
            img_array[:split_height, split_width:],
            img_array[split_height:, :split_width],
            img_array[split_height:, split_width:]
        ]
        new_img_array = np.empty_like(img_array)
        new_img_array[:split_height, :split_width] = img_blocks[order[0]]
        new_img_array[:split_height, split_width:] = img_blocks[order[1]]
        new_img_array[split_height:, :split_width] = img_blocks[order[2]]
        new_img_array[split_height:, split_width:] = img_blocks[order[3]]
        return new_img_array

    @staticmethod
    def _cube_to_pano(results, key, ignore_index=255):
        if len(results[key].shape) == 2:
            h, w = results[key].shape
            img = np.full((h, w * 6), fill_value=ignore_index, dtype=np.uint8)
            img[:, :w] = results[key]  # cube image
            img = img[:, :, np.newaxis]
            img = py360convert.c2e(img, h=2*h, w=4*w, cube_format='horizon', mode='nearest')
            img = img[:, :, 0]
        else:
            h, w, _ = results[key].shape
            img = np.full((h, w * 6, 3), fill_value=ignore_index, dtype=np.uint8)
            img[:, :w] = results[key]  # cube image
            img = py360convert.c2e(img, h=2*h, w=4*w, cube_format='horizon', mode='bilinear')
        h_start = int(0.5 * h)
        h_end = int(0.5 * h + h)
        w_start = int(1.5 * w)
        w_end = int(1.5 * w + w)
        img = img[h_start:h_end, w_start:w_end]
        return img

    def transform(self, results: dict) -> dict:
        if self.shuffle:
            order = [0, 1, 2, 3]
            rd.shuffle(order)
            results['img'] = self._shuffle(results, 'img', order)
            results['gt_seg_map'] = self._shuffle(results, 'gt_seg_map', order)
        pad_h = self.crop_size[0] - results['img'].shape[0]
        pad_w = self.crop_size[1] - results['img'].shape[1]
        results['img'] = np.pad(results['img'], ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=(0, 0))
        results['gt_seg_map'] = np.pad(results['gt_seg_map'], ((0, pad_h), (0, pad_w)), 'constant', constant_values=(self.ignore_index, self.ignore_index))
        results['img_shape'] = results['img'].shape[:2]
        results['img'] = self._cube_to_pano(results, 'img', ignore_index=0).astype(np.uint8)
        results['gt_seg_map'] = self._cube_to_pano(results, 'gt_seg_map', ignore_index=self.ignore_index).astype(np.uint8)
        return results


def parse_args():
    parser = ArgumentParser(description='Add erp to coco-stuff dataset')
    parser.add_argument('--mode', default='train', help='create erp for train or val set. should be "train" or "val". default: train')
    parser.add_argument('--num_workers', type=int, default=64, help='number of workers to process images. default: 4')
    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the image patches. default: False')
    args = parser.parse_args()
    return args
# def parse_args():
#     parser = ArgumentParser(description='Add erp to coco-stuff dataset')
#     parser.add_argument('--mode', default='train', help='create erp for train or val set. should be "train" or "val". default: train')
#     parser.add_argument('--num_workers', type=int, default=64, help='number of workers to process images. default: 64')
#     # 默认值改成 True
#     parser.add_argument('--shuffle', action='store_false', help='disable shuffle, default: True')
#     args = parser.parse_args()
#     # 这里让 shuffle 默认是 True，现在是有shuffle参数就是把shuffle关掉，方便我在平台上用
#     if not hasattr(args, 'shuffle'):
#         args.shuffle = True
#     return args

# xtt

# xtt 9.16 add 跳过已经处理的文件
# def process_image(i):
#     img_path, ann_path = img_paths[i], ann_paths[i]
#     results = dict(img_path=img_path, seg_map_path=ann_path, reduce_zero_label=False, seg_fields=[])
#     results = load_img(results)
#     results = load_ann(results)
#     results = resize(results)
#     results = crop(results)
#     results = pano(results)
#     img = results['img']
#     ann = results['gt_seg_map']
#     if args.shuffle:
#         cv2.imwrite(os.path.join(img_dir_out, 'erp_shuffle_' + os.path.basename(img_path)), img)
#         cv2.imwrite(os.path.join(ann_dir_out, 'erp_shuffle_' + os.path.basename(ann_path)), ann)
#     else:
#         cv2.imwrite(os.path.join(img_dir_out, 'erp_no_shuffle_' + os.path.basename(img_path)), img)
#         cv2.imwrite(os.path.join(ann_dir_out, 'erp_no_shuffle_' + os.path.basename(ann_path)), ann)

def process_image(i):
    img_path, ann_path = img_paths[i], ann_paths[i]

    if args.shuffle:
        out_img = os.path.join(img_dir_out, 'erp_shuffle_' + os.path.basename(img_path))
        out_ann = os.path.join(ann_dir_out, 'erp_shuffle_' + os.path.basename(ann_path))
    else:
        out_img = os.path.join(img_dir_out, 'erp_no_shuffle_' + os.path.basename(img_path))
        out_ann = os.path.join(ann_dir_out, 'erp_no_shuffle_' + os.path.basename(ann_path))

    # 如果文件已存在，直接跳过
    if os.path.exists(out_img) and os.path.exists(out_ann):
        return

    # 正常处理
    results = dict(img_path=img_path, seg_map_path=ann_path, reduce_zero_label=False, seg_fields=[])
    results = load_img(results)
    results = load_ann(results)
    results = resize(results)
    results = crop(results)
    results = pano(results)
    img = results['img']
    ann = results['gt_seg_map']

    cv2.imwrite(out_img, img)
    cv2.imwrite(out_ann, ann)
# xtt



if __name__ == '__main__':

    args = parse_args()
    num_workers = args.num_workers
    resolution = (640, 640)
    img_dir = f'./data/coco_stuff164k/images/{args.mode}2017'
    ann_dir = f'./data/coco_stuff164k/annotations/{args.mode}2017'
    img_paths = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir)])
    ann_paths = sorted([os.path.join(ann_dir, x) for x in os.listdir(ann_dir) if x.endswith('_labelTrainIds.png')])
    load_img = TRANSFORMS.build(dict(type='LoadImageFromFile'))
    load_ann = TRANSFORMS.build(dict(type='LoadAnnotations'))
    resize = TRANSFORMS.build(dict(type='ResizeShortestEdge', scale=resolution, max_size=1e7))
    crop = TRANSFORMS.build(dict(type='RandomCrop', crop_size=resolution, cat_max_ratio=1.0))

    if args.shuffle:
        img_dir_out = img_dir + '_erp_shuffle'
        ann_dir_out = ann_dir + '_erp_shuffle'
        os.makedirs(img_dir_out, exist_ok=True)
        os.makedirs(ann_dir_out, exist_ok=True)
        pano = TRANSFORMS.build(dict(type='PanoTrans', shuffle=True, crop_size=resolution))
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_image, range(len(img_paths))), total=len(img_paths)))
    else:
        img_dir_out = img_dir + '_erp_no_shuffle'
        ann_dir_out = ann_dir + '_erp_no_shuffle'
        os.makedirs(img_dir_out, exist_ok=True)
        os.makedirs(ann_dir_out, exist_ok=True)
        pano = TRANSFORMS.build(dict(type='PanoTrans', shuffle=False, crop_size=resolution))
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_image, range(len(img_paths))), total=len(img_paths)))
