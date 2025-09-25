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
    ERP / RERP（随机打乱 + ERP） 统一实现
    mode='erp' : 只做 ERP
    mode='rerp': 先切块随机打乱，再做 ERP

    Args:
        mode (str): 'erp' 或 'rerp'
        shuffle (bool): 兼容旧参；当 mode='rerp' 时启用
        p_shuffle (float): 发生打乱的概率（0~1）
        grid (int): 网格数（2=>2x2），仅在未设置 grid_choices 时使用
        grid_choices (List[int] | None): 若给定，每次随机选一个网格数，如 [2,3]
        fov_deg (float): 假定 pinhole 的水平 FoV
        crop_size (Tuple[int,int] | None): 输出尺寸 (H,W)，None 表示保持输入
        strength (float): ERP 在纬向的缩放，1.0~1.2
        ignore_index (int): 标注在 FoV 外的填充值
    """
    def __init__(self,
                 mode='erp',
                 shuffle=False,
                 p_shuffle=1.0,
                 grid=2,
                 grid_choices=None,
                 fov_deg=90.0,
                 crop_size=None,
                 strength=1.0,
                 ignore_index=255):
        assert mode in ('erp', 'rerp')
        self.mode = mode
        self.shuffle = bool(shuffle)
        self.p_shuffle = float(p_shuffle)
        self.grid = int(grid)
        self.grid_choices = list(grid_choices) if grid_choices is not None else None
        self.fov_deg = float(fov_deg)
        self.crop_size = crop_size  # (H, W)
        self.strength = float(strength)
        self.ignore_index = int(ignore_index)

        assert self.grid >= 1
        if self.grid_choices is not None:
            assert all(g >= 1 for g in self.grid_choices)

    def transform(self, results):
        img = results['img']
        gt  = results.get('gt_seg_map', None)

        # ---------- Step 0: R（可选打乱） ----------
        if self.mode == 'rerp' and self.shuffle and (np.random.rand() < self.p_shuffle):
            g = np.random.choice(self.grid_choices) if self.grid_choices else self.grid
            img, gt = self._grid_shuffle_both(img, gt, g)

        # ---------- Step 1: ERP 投影 ----------
        H_in, W_in = img.shape[:2]
        H_out, W_out = (self.crop_size if self.crop_size else (H_in, W_in))
        map_x, map_y, _ = self._build_inverse_map(H_out, W_out, W_in, H_in)

        # 图像：双线性，FoV 外填黑
        img_out = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # 标注：最近邻，FoV 外填 ignore_index
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
        """对 img 和 gt 用同一随机置换进行切块打乱。"""
        img_out = self._grid_shuffle(img, grid, is_mask=False)
        gt_out  = None
        if gt is not None:
            gt_out = self._grid_shuffle(gt, grid, is_mask=True)
        return img_out, gt_out

    def _grid_shuffle(self, arr, grid, is_mask=False):
        H, W = arr.shape[:2]
        ph, pw = H // grid, W // grid
        tiles, boxes = [], []
        for i in range(grid):
            for j in range(grid):
                y0, y1 = i * ph, (i + 1) * ph if i < grid - 1 else H
                x0, x1 = j * pw, (j + 1) * pw if j < grid - 1 else W
                tiles.append(arr[y0:y1, x0:x1].copy())
                boxes.append((y0, y1, x0, x1))
        order = np.random.permutation(len(tiles))
        tiles = [tiles[k] for k in order]

        out = np.empty_like(arr)
        k = 0
        for i in range(grid):
            for j in range(grid):
                y0, y1 = i * ph, (i + 1) * ph if i < grid - 1 else H
                x0, x1 = j * pw, (j + 1) * pw if j < grid - 1 else W
                tile = tiles[k]
                interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
                tile = cv2.resize(tile, (x1 - x0, y1 - y0), interpolation=interp)
                out[y0:y1, x0:x1] = tile
                k += 1
        return out

    def _build_inverse_map(self, H_out, W_out, W_in, H_in):
        """
        用 pinhole 的 FoV 子域生成 ERP 输出（把 FoV 子域铺满整个输出画布）
        """
        xs = (np.arange(W_out, dtype=np.float32)[None, :] + 0.5) / W_out  # [1, W]
        ys = (np.arange(H_out, dtype=np.float32)[:, None] + 0.5) / H_out  # [H, 1]

        fov_x = math.radians(self.fov_deg)
        aspect = W_in / float(H_in)
        fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)

        alpha = fov_x / 2.0   # 水平半视场
        beta  = fov_y / 2.0   # 垂直半视场

        lam = (xs * (2.0 * alpha)) - alpha              # [-alpha, +alpha]
        phi = (ys * (2.0 * beta)) - beta                # [-beta, +beta]
        phi = phi * self.strength

        # 透视投影推导：x_p = tan(lam), y_p = tan(phi)/cos(lam)
        x_p = np.tan(lam)
        cos_lam = np.cos(lam)
        cos_lam_safe = np.clip(cos_lam, 1e-6, None)
        y_p = np.tan(phi) / cos_lam_safe

        tanx = math.tan(alpha)
        tany = math.tan(beta)
        u = (x_p / tanx + 1.0) * 0.5     # (1, W)
        v = (y_p / tany + 1.0) * 0.5     # (H, W)

        # 扩展 u 到 (H, W)
        if u.ndim == 2 and u.shape[0] == 1:
            u = np.repeat(u, H_out, axis=0)
        elif u.ndim == 1:
            u = np.tile(u[None, :], (H_out, 1))

        map_x = (u * W_in).astype(np.float32)           # (H, W)
        map_y = (v * H_in).astype(np.float32)           # (H, W)

        valid = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        map_x[~valid] = -1.0
        map_y[~valid] = -1.0
        return map_x, map_y, valid

    def __repr__(self):
        return (f'{self.__class__.__name__}(mode={self.mode}, shuffle={self.shuffle}, '
                f'p_shuffle={self.p_shuffle}, grid={self.grid}, '
                f'fov_deg={self.fov_deg}, crop_size={self.crop_size}, '
                f'strength={self.strength}, ignore_index={self.ignore_index}, '
                f'grid_choices={self.grid_choices})')
