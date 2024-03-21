import os, shutil

if __name__ == "__main__":
    src_pth = "config"
    target_path = "corl/config"

    if os.path.exists(src_pth):
        if os.path.exists(target_path) and os.path.isdir(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(src_pth, target_path)
