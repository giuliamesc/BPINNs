import os
import shutil

src_wd = os.path.join(os.getcwd(), "src")
for folder in os.listdir(src_wd):
    if folder[-3:] == ".py": continue
    folder_path = os.path.join(src_wd, folder)
    cache = os.path.join(folder_path, "__pycache__")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print(cache)
    if folder == "networks":
        folder_path = os.path.join(folder_path, "equations")
        cache = os.path.join(folder_path, "__pycache__")
        if os.path.isdir(cache):
            shutil.rmtree(cache)
            print(cache)
    if folder == "setup":
        folder_path = os.path.join(folder_path, "config")
        cache = os.path.join(folder_path, "__pycache__")
        if os.path.isdir(cache):
            shutil.rmtree(cache)
            print(cache)

outs_wd = os.path.join(os.getcwd(), "outs")
for folder in os.listdir(outs_wd):
    if folder == ".gitkeep": continue
    folder_path = os.path.join(outs_wd, folder)
    trash = os.path.join(folder_path,"trash")
    if os.path.isdir(trash):
        shutil.rmtree(trash)
        print(trash)

data_wd = os.path.join(os.getcwd(), "data")
trash = os.path.join(data_wd,"trash")
if os.path.isdir(trash):
    shutil.rmtree(trash)
    print(trash)

