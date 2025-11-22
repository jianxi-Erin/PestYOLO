import gdown
import os
import tarfile
import subprocess

def download_and_extract_dataset():
    # 数据集保存路径
    root = "/IP102_v1.1"

    # Google Drive 文件夹链接
    folder_url = "https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo"

    # 判断数据集是否已经存在
    if os.path.exists(root) and os.listdir(root):
        print(f"数据集已存在，跳过下载解压: {root}")
        return

    print(f"开始下载数据集到: {root}")
    
    # 使用gdown下载文件夹
    try:
        # 方法1: 使用gdown的Python API
        gdown.download_folder(url=folder_url, output=root, quiet=False)
    except Exception as e:
        print(f"使用Python API下载失败: {e}")
        print("尝试使用命令行方式下载...")
        # 方法2: 使用命令行方式
        try:
            result = subprocess.run([
                'gdown', '--folder', folder_url, '-O', root
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"命令行下载失败: {result.stderr}")
                return
        except Exception as e:
            print(f"命令行方式也失败了: {e}")
            return

    print("----开始解压数据集----")

    # 压缩包列表
    tar_files = [
        os.path.join(root, "Classification/ip102_v1.1.tar"),
        os.path.join(root, "Detection/VOC2007/Annotations.tar"),
        os.path.join(root, "Detection/VOC2007/JPEGImages.tar")
    ]

    # 解压函数
    def extract_tar(tar_path, output_dir=None):
        if output_dir is None:
            output_dir = os.path.dirname(tar_path)
        
        # 检查tar文件是否存在
        if not os.path.exists(tar_path):
            print(f"❌ 压缩文件不存在: {tar_path}")
            return False
            
        print(f"解压: {tar_path} → {output_dir}")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=output_dir)
            print(f"✅ 完成: {tar_path}")
            return True
        except Exception as e:
            print(f"❌ 解压失败 {tar_path}: {e}")
            return False

    # 创建必要的目录结构
    required_dirs = [
        os.path.join(root, "Classification"),
        os.path.join(root, "Detection/VOC2007")
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # 遍历解压
    success_count = 0
    for tar_file in tar_files:
        if extract_tar(tar_file):
            success_count += 1

    print(f"\n解压完成: {success_count}/{len(tar_files)} 个文件成功解压")
    
    # 检查最终目录结构
    print("\n数据集结构检查:")
    check_dirs = [
        root,
        os.path.join(root, "Classification"),
        os.path.join(root, "Detection"),
        os.path.join(root, "Detection/VOC2007")
    ]
    
    for check_dir in check_dirs:
        if os.path.exists(check_dir):
            file_count = len([f for f in os.listdir(check_dir) if os.path.isfile(os.path.join(check_dir, f))])
            dir_count = len([d for d in os.listdir(check_dir) if os.path.isdir(os.path.join(check_dir, d))])
            print(f"📁 {check_dir}: {file_count} 个文件, {dir_count} 个子目录")
        else:
            print(f"❌ 目录不存在: {check_dir}")

if __name__ == "__main__":
    download_and_extract_dataset()