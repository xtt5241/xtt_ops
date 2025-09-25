# -*- coding: utf-8 -*-
# file: tools/dataset_converters/pano_trans.py
import math
import numpy as np
import cv2

from mmseg.registry import TRANSFORMS
try:
    from mmcv.transforms import BaseTransform  # OpenMMLab 2.x
except Exception:
    from mmseg.datasets.transforms import BaseTransform  # 旧版兜底


@TRANSFORMS.register_module()
class PanoTrans(BaseTransform):
    """
    ERP / RERP（随机打乱 + ERP）统一实现
    mode='erp' : 只做 ERP
    mode='rerp': 先切块随机打乱，再做 ERP
    """
    def __init__(self,
                 mode='erp',
                 shuffle=False,
                 p_shuffle=1.0,
                 grid=2,
                 grid_choices=None,
                 fov_deg=90.0,
                 out_size=None,
                 strength=1.0,
                 ignore_index=255):
        assert mode in ('erp', 'rerp')
        self.mode = mode
        self.shuffle = bool(shuffle)
        self.p_shuffle = float(p_shuffle)
        self.grid = int(grid)
        self.grid_choices = list(grid_choices) if grid_choices is not None else None
        self.fov_deg = float(fov_deg)
        self.out_size = out_size  # (H, W)
        self.strength = float(strength)
        self.ignore_index = int(ignore_index)

        assert self.grid >= 1
        if self.grid_choices is not None:
            assert all(int(g) >= 1 for g in self.grid_choices)

    def transform(self, results):
        img = results['img']
        gt  = results.get('gt_seg_map', None)

        # ---------- Step 0: R（可选打乱，同一置换） ----------
        if self.mode == 'rerp' and self.shuffle and (np.random.rand() < self.p_shuffle):
            g = int(np.random.choice(self.grid_choices)) if self.grid_choices else self.grid
            img, gt = self._grid_shuffle_both(img, gt, g)

        # ---------- Step 1: ERP 投影（同一映射） ----------
        H_in, W_in = img.shape[:2]
        H_out, W_out = (self.out_size if self.out_size else (H_in, W_in))
        map_x, map_y, _ = self._build_inverse_map(H_out, W_out, W_in, H_in)

        # 图像：双线性；FoV 外填黑
        img_out = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # 标注：最近邻；FoV 外填 ignore_index
        if gt is not None:
            if gt.ndim == 3 and gt.shape[2] == 1:
                gt = gt[..., 0]
            gt_out = cv2.remap(
                gt, map_x, map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_index
            )
            results['gt_seg_map'] = gt_out

        results['img'] = img_out
        return results

    # ---- helpers ----
    def _grid_shuffle_both(self, img, gt, grid):
        """对 img 与 gt 使用同一随机置换与同一 boxes；鲁棒避免空 tile。"""
        H, W = img.shape[:2]
        g = int(min(max(grid, 1), max(1, min(H, W))))  # 不超过最小边
        img_tiles, boxes = self._split_tiles(img, g)
        if not img_tiles:
            return img, gt

        gt_out = None
        if gt is not None:
            gt_tiles, boxes_gt = self._split_tiles(gt, g)
            # 任一异常则退化为不打乱，避免错位
            if (not gt_tiles) or (len(gt_tiles) != len(img_tiles)) or (boxes_gt != boxes):
                return img, gt

        order = np.random.permutation(len(img_tiles))
        img_out = self._merge_tiles(img.shape, [img_tiles[k] for k in order], boxes, is_mask=False)
        if gt is not None:
            gt_out = self._merge_tiles(gt.shape,  [gt_tiles[k]  for k in order], boxes, is_mask=True)
        return img_out, gt_out

    def _split_tiles(self, arr, grid):
        """按 grid×grid 划分为非空 tile；返回 tiles 与对应 boxes。"""
        H, W = arr.shape[:2]
        if H < 2 or W < 2 or grid <= 1:
            return [arr.copy()], [(0, H, 0, W)]
        ys = np.linspace(0, H, grid + 1, dtype=int)
        xs = np.linspace(0, W, grid + 1, dtype=int)
        tiles, boxes = [], []
        for i in range(grid):
            for j in range(grid):
                y0, y1 = ys[i], ys[i + 1]
                x0, x1 = xs[j], xs[j + 1]
                if y1 <= y0 or x1 <= x0:
                    continue
                tiles.append(arr[y0:y1, x0:x1].copy())
                boxes.append((y0, y1, x0, x1))
        if not tiles:
            return [arr.copy()], [(0, H, 0, W)]
        return tiles, boxes

    def _merge_tiles(self, shape, tiles, boxes, is_mask=False):
        """把打乱后的 tiles 放回 boxes；必要时 resize。"""
        out = np.empty(shape, dtype=tiles[0].dtype)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        k = 0
        for (y0, y1, x0, x1) in boxes:
            tile = tiles[k]; k += 1
            dh, dw = (y1 - y0), (x1 - x0)
            if tile.shape[0] != dh or tile.shape[1] != dw:
                tile = cv2.resize(tile, (dw, dh), interpolation=interp)
            out[y0:y1, x0:x1] = tile
        return out

    def _build_inverse_map(self, H_out, W_out, W_in, H_in):
        """把 pinhole FoV 子域铺满输出画布（用于 ERP/RERP）。"""
        xs = (np.arange(W_out, dtype=np.float32)[None, :] + 0.5) / W_out
        ys = (np.arange(H_out, dtype=np.float32)[:, None] + 0.5) / H_out

        fov_x = math.radians(self.fov_deg)
        aspect = W_in / float(H_in)
        fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)

        alpha = fov_x / 2.0
        beta  = fov_y / 2.0

        lam = (xs * (2.0 * alpha)) - alpha
        phi = (ys * (2.0 * beta)) - beta
        phi = phi * self.strength

        x_p = np.tan(lam)
        cos_lam = np.cos(lam)
        y_p = np.tan(phi) / np.clip(cos_lam, 1e-6, None)

        u = (x_p / math.tan(alpha) + 1.0) * 0.5
        v = (y_p / math.tan(beta)  + 1.0) * 0.5
        if u.ndim == 2 and u.shape[0] == 1:
            u = np.repeat(u, H_out, axis=0)

        map_x = (u * W_in).astype(np.float32)
        map_y = (v * H_in).astype(np.float32)

        valid = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        map_x[~valid] = -1.0
        map_y[~valid] = -1.0
        return map_x, map_y, valid

    def __repr__(self):
        return (f'{self.__class__.__name__}(mode={self.mode}, shuffle={self.shuffle}, '
                f'p_shuffle={self.p_shuffle}, grid={self.grid}, '
                f'fov_deg={self.fov_deg}, out_size={self.out_size}, '
                f'strength={self.strength}, ignore_index={self.ignore_index}, '
                f'grid_choices={self.grid_choices})')
