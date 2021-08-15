import cv2
import os
import numpy as np
import logging
logging.basicConfig(format='%(filename)s-[line:%(lineno)d]-%(levelname)s: %(message)s',level=logging.INFO)

# coding: utf-8
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img
def cv_write(img_path,im):
    os.makedirs(os.path.dirname(img_path),exist_ok=True)
    cv2.imencode('.png',im)[1].tofile(img_path+".png")


def download_img(img,out_path,str,downloadImg=True,do_log=False):
    if True == os.path.exists(out_path):
        remove_dirs(out_path)
    if downloadImg:
        img_path = out_path  + str
        cv_write(img_path, img)
        if do_log == True:
            logging.info("[SUCESS] imwrite :",img_path)

def get_file(root_path,all_files):
    '''
    递归函数，遍历该文档目录和子目录下的所有文件，获取其path
    '''
    #all_files = []
    files = os.listdir(root_path)
    for file in files:
        if not os.path.isdir(root_path + '/' + file):   # not a dir
            #if file.endswith(farmat):
            all_files.append(root_path + '/' + file)
        else:  # is a dir
            get_file((root_path+'/'+file),all_files)
    return all_files

def remove_dirs(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))