import os
import shutil

def print_path(full_path):
    index = full_path.find("pacs_bpinns")
    small_path = full_path[index:]
    print(small_path)

src_wd = os.path.join(os.getcwd(), "src")
for folder in os.listdir(src_wd):
    if folder[-3:] == ".py": continue
    folder_path = os.path.join(src_wd, folder)
    cache = os.path.join(folder_path, "__pycache__")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print_path(cache)

outs_wd = os.path.join(os.getcwd(), "outs")
for folder in os.listdir(outs_wd):
    if folder == ".gitkeep": continue
    folder_path = os.path.join(outs_wd, folder)
    for folder in os.listdir(folder_path):
        if folder == ".gitkeep": continue
        folder_path = os.path.join(folder_path, folder)
        trash = os.path.join(folder_path, "trash")
        if os.path.isdir(trash):
            shutil.rmtree(trash)
            print_path(trash)

data_wd = os.path.join(os.getcwd(), "data")
for folder in os.listdir(outs_wd):
    if folder == ".gitkeep": continue
    folder_path = os.path.join(data_wd, folder)  
    trash = os.path.join(folder_path, "trash")
    if os.path.isdir(trash):
        shutil.rmtree(trash)
        print_path(trash)

