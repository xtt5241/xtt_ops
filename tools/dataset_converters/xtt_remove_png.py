import os

def remove_png_files(root_dir):
    """递归删除 root_dir 下的所有 .png 文件"""
    removed_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                removed_files.append(file_path)
                print(f"Deleted: {file_path}")
    print(f"\n共删除 {len(removed_files)} 个 .png 文件")
    return removed_files

if __name__ == "__main__":
    # 修改为你的数据集路径
    dataset_path = "/opt/data/private/xtt_OPS/OPS/data/coco_stuff164k/images/val2017"
    remove_png_files(dataset_path)
