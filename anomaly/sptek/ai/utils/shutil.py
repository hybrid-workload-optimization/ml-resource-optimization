import os
import shutil

def copy(src, dst, rm_dst=False):
    if rm_dst and os.path.exists(dst):
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            os.remove(dst)
   
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy(src, dst)