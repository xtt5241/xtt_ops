import os
import cv2
from mmseg.registry import TRANSFORMS

# 关键：显式导入，触发 PanoTrans 注册到 mmseg 的 TRANSFORMS
from pano_trans import PanoTrans  # noqa: F401

def main():
    img_path = "/opt/data/private/xtt_OPS/OPS/tools/dataset_converters/000000000036.jpg"
    out_path = "/opt/data/private/xtt_OPS/OPS/tools/dataset_converters/000000000036_out.jpg"

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert img is not None, f"failed to read: {img_path}"

    results = {'img_path': img_path, 'img': img}

    # pano = TRANSFORMS.build(dict(
    #     type='PanoTrans',
    #     mode='erp',
    #     fov_deg=90.0,
    #     strength=1.0,
    #     out_size=(640, 640),
    #     ignore_index=255
    # ))

    pano = TRANSFORMS.build(dict(
        type='PanoTrans',
        mode='rerp',
        shuffle=True,          # 开打乱
        p_shuffle=1.0,         # 打乱概率；训练可调 0.5~1.0
        grid=2,                # 2x2；或用 grid_choices 在多种网格间随机
        # grid_choices=[2,3],  # 在 2x2 / 3x3 随机
        fov_deg=90.0,
        strength=1.0,
        out_size=(640, 640),
        ignore_index=255
    ))


    results = pano(results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, results['img'])
    print("done ->", out_path)





if __name__ == "__main__":
    main()
