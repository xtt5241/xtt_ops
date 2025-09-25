#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open mIoU evaluation (WordNet Path), with closed-set self-check.
修正点：不再对 GT 做 clip，严格忽略无效 GT（与 mmseg 官方评测一致）。
输出：
- 闭集 mIoU (S=I) ：no-shift / shift(-1) / best
- Open mIoU（first-sense / max-sense，在 best 对齐方式上）
默认类别顺序严格对齐你的 config vocabulary。
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# ========= 确保 nltk 可用（清华源），你已装好 wordnet，这里不强制下载 =========
def ensure_nltk():
    try:
        import nltk  # noqa
        print(f"[INFO] 已检测到 nltk (version={nltk.__version__})")
        return nltk
    except ImportError:
        print("[WARN] 未检测到 nltk，正在安装 (清华源)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-i",
            "https://pypi.tuna.tsinghua.edu.cn/simple", "nltk"
        ])
        import nltk  # noqa
        print(f"[OK] 已安装 nltk (version={nltk.__version__})")
        return nltk

nltk = ensure_nltk()
from nltk.corpus import wordnet as wn  # noqa

# 你的离线 wordnet 目录（如果存在则追加）
NLTK_PATH = "/opt/data/private/xtt_OPS/OPS/tools/open_miou"
if os.path.isdir(NLTK_PATH):
    nltk.data.path.append(NLTK_PATH)

# ========= IO & 工具 =========
def read_classes(classes_file: Optional[str], fallback: Optional[List[str]] = None) -> List[str]:
    if classes_file and Path(classes_file).is_file():
        labels = [line.strip() for line in open(classes_file, "r", encoding="utf-8") if line.strip()]
        if labels:
            return labels
    if fallback:
        return fallback
    raise FileNotFoundError("[ERROR] 未提供有效的 classes 文件，且无 fallback 列表。")

def list_pngs(dir_path: str) -> List[Path]:
    return sorted([p for p in Path(dir_path).glob("*.png")])

def match_pairs(pred_dir: str, gt_dir: str) -> List[Tuple[Path, Path]]:
    pred_files = {p.name: p for p in list_pngs(pred_dir)}
    pairs = []
    for gt in list_pngs(gt_dir):
        if gt.name in pred_files:
            pairs.append((pred_files[gt.name], gt))
    return pairs

def load_mask(p: Path) -> np.ndarray:
    arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)

# ========= WordNet 相似度（Path）=========
# 轻量词形规范：objects->object, shelving->shelf, counter->countertop(若无则回退counter)
LEMMA_CANON = {
    "objects": "object",
    "shelving": "shelf",
    "counter": "countertop",
}

def normalize_variants(label: str) -> List[str]:
    raw = label.strip().lower()
    raw = LEMMA_CANON.get(raw, raw)
    cands = [raw, raw.replace("_", " "), raw.replace("-", " ")]
    # morphy 名词词形还原
    for w in list(cands):
        lm = wn.morphy(w, wn.NOUN)
        if lm and lm not in cands:
            cands.append(lm)
    if " " in raw:
        parts = raw.split()
        tail2 = " ".join(parts[-2:])
        if tail2 not in cands:
            cands.append(tail2)
        join_ = "_".join(parts)
        if join_ not in cands:
            cands.append(join_)
    # 去重保序
    seen, out = set(), []
    for x in cands:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out

def collect_synsets(label: str) -> List:
    syns = []
    for q in normalize_variants(label):
        syns += wn.synsets(q, pos=wn.NOUN)
    # 'countertop' 没有就回退 'counter'
    if label.strip().lower() == "counter" and not syns:
        syns += wn.synsets("counter", pos=wn.NOUN)
    uniq, seen = [], set()
    for s in syns:
        if s.name() not in seen:
            uniq.append(s)
            seen.add(s.name())
    if not uniq:
        return [wn.synset("entity.n.01")]
    return uniq

def build_similarity_matrix(labels: List[str], mode: str = "max") -> np.ndarray:
    """
    mode:
      - 'max': 两类所有义项对取最大 path 相似度（推荐）
      - 'first': 仅第一义项（baseline）
      - 'identity': 单位阵（闭集 mIoU 自检）
    """
    C = len(labels)
    S = np.zeros((C, C), dtype=np.float64)
    if mode == "identity":
        np.fill_diagonal(S, 1.0)
        return S
    syns_list = [collect_synsets(n) for n in labels]
    for i in range(C):
        S[i, i] = 1.0
        for j in range(i + 1, C):
            if mode == "first":
                s1 = syns_list[i][0] if syns_list[i] else None
                s2 = syns_list[j][0] if syns_list[j] else None
                sim = (s1.path_similarity(s2) or 0.0) if (s1 and s2) else 0.0
            else:  # 'max'
                best = 0.0
                for s1 in syns_list[i]:
                    for s2 in syns_list[j]:
                        v = s1.path_similarity(s2) or 0.0
                        if v > best:
                            best = v
                sim = best
            S[i, j] = S[j, i] = float(sim)
    return S

# ========= 累积混淆矩阵（关键修正：不 clip GT，严格忽略无效 GT）=========
def accumulate_confusion(
    pairs: List[Tuple[Path, Path]],
    C: int,
    ignore_index: int,
    shift_pred: int = 0,
    reduce_zero_label: bool = False,   # 新增
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    返回：
      - conf:  CxC 混淆矩阵 (行=GT, 列=Pred)
      - gt_sz: 每类 GT 像素数（仅统计合法 GT）
      - pd_sz: 每类 Pred 像素数（对应合法 GT 掩码后）
      - stats: 诊断信息（无效 GT 像素总数等）
    注意：GT 只保留 (0 <= gt < C) 且 (gt != ignore_index)；其余全部忽略。
    """
    conf = np.zeros((C, C), dtype=np.int64)
    gt_sz = np.zeros((C,), dtype=np.int64)
    pd_sz = np.zeros((C,), dtype=np.int64)

    gt_oor_total = 0  # GT out-of-range 或 ignore 的像素总数（被忽略）
    total_pixels_considered = 0

    for pred_p, gt_p in pairs:
        pred = load_mask(pred_p)
        gt = load_mask(gt_p)

        # ←← 在尺寸判断之前或之后都可以，这里放在之后也行
        if reduce_zero_label:
            gt = apply_reduce_zero_label(gt, ignore_index=ignore_index)

        if pred.shape != gt.shape:
            continue

        if pred.shape != gt.shape:
            continue

        # 仅保留合法 GT：0..C-1 且 != ignore_index
        mask_valid = (gt != ignore_index) & (gt >= 0) & (gt < C)
        if not np.any(mask_valid):
            # 全无合法 GT，整张跳过
            continue

        g = gt[mask_valid].ravel()
        p = pred[mask_valid].ravel()

        if shift_pred == -1:
            p = p - 1
        # 预测仍可 clip，因为 head 通常就是 0..C-1，但为稳健还是限制一下
        p = np.clip(p, 0, C - 1)

        idx = g * C + p
        binc = np.bincount(idx, minlength=C * C)
        conf += binc.reshape(C, C)

        gt_sz += np.bincount(g, minlength=C)
        pd_sz += np.bincount(p, minlength=C)

        # 统计被忽略的 GT 像素（仅用于诊断）
        gt_oor_total += (gt.size - mask_valid.sum())
        total_pixels_considered += mask_valid.sum()

    stats = {
        "gt_ignored_pixels": int(gt_oor_total),
        "valid_gt_pixels": int(total_pixels_considered),
    }
    return conf, gt_sz, pd_sz, stats

# ========= 计算 mIoU =========
def closed_miou_from_conf(conf: np.ndarray) -> float:
    inter = np.diag(conf).astype(np.float64)
    gt_sum = conf.sum(axis=1).astype(np.float64)
    pd_sum = conf.sum(axis=0).astype(np.float64)
    union = gt_sum + pd_sum - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=(union > 0))
    valid = (gt_sum > 0)   # ← 关键：只对 GT 出现过的类取均值（与 mmseg 一致）
    return float(iou[valid].mean()) if np.any(valid) else 0.0


def apply_reduce_zero_label(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """模仿 mmseg 的 reduce_zero_label=True：
       - 原始 0 类（通常是 background/others）置成 ignore_index
       - 其它有效类整体减 1，使得标签范围从 1..C 变成 0..C-1
    """
    m = mask.copy()
    zero = (m == 0)
    m[zero] = ignore_index
    keep = (m != ignore_index)
    m[keep] = m[keep] - 1
    return m

def per_class_iou(conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inter = np.diag(conf).astype(np.float64)
    gt_sum = conf.sum(axis=1).astype(np.float64)
    pd_sum = conf.sum(axis=0).astype(np.float64)
    union = gt_sum + pd_sum - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=(union > 0))
    valid = (gt_sum > 0)  # 与 mmseg 一致：只统计 GT 出现过的类
    return iou, inter, gt_sum, union


def open_miou_from_conf(conf: np.ndarray, S: np.ndarray) -> float:
    n = conf.astype(np.float64)
    TP = (S * n).sum(axis=1)
    FN = ((1.0 - S) * n).sum(axis=1)
    FP = ((1.0 - S.T) * n.T).sum(axis=1)
    denom = TP + FP + FN
    iou = np.divide(TP, denom, out=np.zeros_like(TP), where=(denom > 0))
    valid = (denom > 0)
    return float(iou[valid].mean()) if np.any(valid) else 0.0

import numpy as np
from PIL import Image

pred = np.array(Image.open("/opt/data/private/xtt_OPS/OPS/work_dirs/ops-vit-b16_m3d-640x640/preds/masks/0a9f30bd318e40de89f71e4bf6987358.png"))
gt   = np.array(Image.open("/opt/data/private/xtt_OPS/OPS/data/matterport3d/val/semantic/0a9f30bd318e40de89f71e4bf6987358.png"))

print("Pred unique:", np.unique(pred)[:20])
print("GT unique:", np.unique(gt)[:20])




# ========= 主逻辑 =========
def main():
    parser = argparse.ArgumentParser(description="Open mIoU (WordNet path) evaluator")
    parser.add_argument("--pred-dir",
                        default="/opt/data/private/xtt_OPS/OPS/work_dirs/ops-vit-b16_m3d-640x640/preds/masks")
    parser.add_argument("--gt-dir",
                        default="/opt/data/private/xtt_OPS/OPS/data/matterport3d/val/semantic")
    parser.add_argument("--classes",
                        default="/opt/data/private/xtt_OPS/OPS/data/matterport3d/classes.txt")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--reduce-zero-label", action="store_true", default=True)

    args = parser.parse_args()

    # 类别（优先 classes.txt；否则与你 config vocabulary 完全一致）
    vocab_from_config = [
        'wall', 'floor', 'chair', 'door', 'table', 'picture', 'furniture', 'objects',
        'window', 'sofa', 'bed', 'sink', 'stairs', 'ceiling', 'toilet', 'mirror',
        'shower', 'bathtub', 'counter', 'shelving'
    ]
    labels = read_classes(args.classes, fallback=vocab_from_config)
    C = len(labels)

    # 匹配
    pairs = match_pairs(args.pred_dir, args.gt_dir)
    if len(pairs) == 0:
        print(f"[ERROR] 在 {args.pred_dir} 和 {args.gt_dir} 没有匹配到同名 .png")
        sys.exit(1)

    # 两种对齐：no-shift / shift(-1)

    conf_0, gt_0, pd_0, st_0 = accumulate_confusion(
        pairs, C, args.ignore_index, shift_pred=0, reduce_zero_label=args.reduce_zero_label
    )
    conf_1, gt_1, pd_1, st_1 = accumulate_confusion(
        pairs, C, args.ignore_index, shift_pred=-1, reduce_zero_label=args.reduce_zero_label
    )


    closed_0 = closed_miou_from_conf(conf_0)
    closed_1 = closed_miou_from_conf(conf_1)

    if closed_1 >= closed_0:
        best_conf = conf_1
        best_align = "shift-1"
        best_closed = closed_1
    else:
        best_conf = conf_0
        best_align = "no-shift"
        best_closed = closed_0

    # WordNet 相似度矩阵
    S_first = build_similarity_matrix(labels, mode="first")
    S_max   = build_similarity_matrix(labels, mode="max")

    open_first = open_miou_from_conf(best_conf, S_first)
    open_max   = open_miou_from_conf(best_conf, S_max)

    # 打印
    print("========== Open mIoU 自检 ==========")
    print(f"闭集 mIoU (S=I, no-shift): {closed_0*100:8.4f}")
    print(f"闭集 mIoU (S=I, shift-1): {closed_1*100:8.4f}")
    print(f"闭集 mIoU (S=I, best={best_align}): {best_closed*100:8.4f}")
    print(f"Open mIoU (first-sense, {best_align}): {open_first*100:8.4f}")
    print(f"Open mIoU (max-sense,   {best_align}): {open_max*100:8.4f}")
    print("===================================")

    # 逐类 IoU 对表
    iou_pc, inter_pc, gt_pc, union_pc = per_class_iou(best_conf)
    print("\n----- Per-class IoU (best alignment: %s) -----" % best_align)
    for i, name in enumerate(labels):
        if gt_pc[i] > 0:  # 只打印 GT 出现过的类
            print(f"{i:2d} {name:<10s} IoU={iou_pc[i]*100:6.2f}  (inter={int(inter_pc[i])}, union={int(union_pc[i])})")
    print("------------------------------------------------\n")


    # 诊断（可选打印，如需更安静可注释）
    ignored = st_0["gt_ignored_pixels"]  # 两次统计相同逻辑，这里取 no-shift 的
    valid   = st_0["valid_gt_pixels"]
    total   = ignored + valid
    if total > 0:
        print(f"[INFO] GT 合法像素: {valid}  忽略像素: {ignored}  (忽略占比 {ignored/total*100:.2f}%)")

    # 保存
    out_file = Path(args.pred_dir) / "open_miou_eval.txt"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write("========== Open mIoU 自检 ==========\n")
        f.write(f"闭集 mIoU (S=I, no-shift): {closed_0*100:.4f}\n")
        f.write(f"闭集 mIoU (S=I, shift-1): {closed_1*100:.4f}\n")
        f.write(f"闭集 mIoU (S=I, best={best_align}): {best_closed*100:.4f}\n")
        f.write(f"Open mIoU (first-sense, {best_align}): {open_first*100:.4f}\n")
        f.write(f"Open mIoU (max-sense,   {best_align}): {open_max*100:.4f}\n")
        f.write("===================================\n")
    print(f"[INFO] 结果已保存到: {out_file}")

if __name__ == "__main__":
    main()
