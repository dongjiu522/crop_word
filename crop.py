import cv2
import sys
import os
import numpy as np
import scipy.signal as singnal
from util import *


#http://www.c-s-a.org.cn/html/2021/2/7819.html
def compute_threshold(img):
    img_max = np.max(img)
    img_min = np.min(img)
    print("img_max = " ,img_max)
    print("img_min = ", img_min)
    img_th = img.copy()
    max_g = 0
    max_g_th = 0
    for th in range(img_min,img_max):
        #img_th[:] = 0
        #cv2.threshold(img, th, 255, cv2.THRESH_BINARY, dst=img_th)
        pixel_size = img_th.size
        more_th_size = np.sum(img >  th)
        low_th_size  = np.sum(img <= th)

        m0 = more_th_size / pixel_size
        n0 = np.mean(img[img > th])

        m1 = low_th_size / pixel_size
        n1 = np.mean(img[img <= th])

        g = m0 * m1 * ((n0 - n1) **2)
        if g > max_g:
            max_g = g
            max_g_th = th
    return max_g_th

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re.astype(dtype=np.uint8)

# 垂直向投影
def vProject(binary):
    h, w = binary.shape
    # 垂直投影
    # 创建 w 长度都为0的数组
    v_projection_array = [0]*w
    for i in range(w):
        for j in range(h):
            if binary[j, i ] == 255:
                v_projection_array[i] += 1
    #v_projection_array_blur = singnal.medfilt(v_projection_array, 9)
    #v_projection_array_blur=v_projection_array_blur.astype(dtype=np.uint8)
    print("[Message] : v_projection_array :",v_projection_array)
    #v_projection_array_blur = moving_average(v_projection_array, 3)
    #print(v_projection_array_blur)
    #v_projection_array_blur = singnal.medfilt(v_projection_array, 5)
    #v_projection_array_blur=v_projection_array_blur.astype(dtype=np.uint8)
    v_projection_array_blur = v_projection_array
    v_projection_img_blur = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(len(v_projection_array_blur)):
        if v_projection_array_blur[i] != 0:
            cv2.line(v_projection_img_blur,(i,h- v_projection_array_blur[i]),(i,h),(255))
    #v_projection_img_blur = ~v_projection_img_blur





    return np.array(v_projection_array_blur),np.array(v_projection_img_blur)

# 水平方向投影
def hProject(binary):
    h, w = binary.shape

    # 水平投影
    h_projection_img = np.zeros(binary.shape, dtype=np.uint8)

    # 创建h长度都为0的数组
    h_projection_array = [0]*h
    for j in range(h):
        for i in range(w):
            if binary[j,i] == 255:
                h_projection_array[j] += 1

    print("[Message] : h_projection_array :",h_projection_array)

    h_projection_img_blur = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(len(h_projection_array)):
        if h_projection_array[i] != 0 :
            cv2.line(h_projection_img_blur,(w- h_projection_array[i],i),(w,i),(255))
    #print("[Message] : h_projection_array :", h_projection_array)
    return np.array(h_projection_array),np.array(h_projection_img_blur)


def crop_word(th,crop_path,downloadImg=True):
    h,w = th.shape
    v_project_array,v_project_img_blur= vProject(th)
    download_img(v_project_img_blur, crop_path, "03-v_projection_img")
    v_project_array_valid = v_project_array[v_project_array > 0]
    v_project_array_max = np.max(v_project_array_valid)
    v_project_array_sum = np.sum(v_project_array_valid)
    v_project_array_median = np.median(v_project_array_valid)
    v_project_array_mean   = v_project_array_sum/ len(v_project_array_valid)
    print("[Message] : v_project_array_median :", v_project_array_median)
    print("[Message] : v_project_array_mean :", v_project_array_mean)
    scale = 0.3
    v_project_array_threshold =  min(v_project_array_mean,v_project_array_median) * scale
    print("[Message] : v_project_array_threshold :", v_project_array_threshold)
    v_project_array[v_project_array < (v_project_array_threshold)] = 0

    v_projection_img = np.zeros(th.shape, dtype=np.uint8)
    for i in range(len(v_project_array)):
        if v_project_array[i] != 0:
            cv2.line(v_projection_img,(i,h- v_project_array[i]),(i,h),(255))

    download_img(v_projection_img, crop_path,"03-v_projection_img_done")

    start = 0
    v_start, v_end = [], []
    position = []

    #应该垂直分割
    #根据水平投影获取垂直分割
    for i in range(len(v_project_array)):
        if v_project_array[i] > 0 and start == 0:
            v_start.append(i)
            start = 1
        if v_project_array[i] ==0 and start == 1:
            v_end.append(i)
            start = 0

    for v_img_index in range(len(v_start)):
        bin_width = abs(v_end[v_img_index] - v_start[v_img_index])
        if bin_width < 20 :
            continue

        cropImg = th[0:h,v_start[v_img_index]:v_end[v_img_index]]

        download_img(cropImg, crop_path,"05-crop_v_" + str(v_img_index))

        #continue
        #应该水平分割
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
        h_project_array, h_project_img= hProject(cropImg)
        #print("[Message] : h_project_array :", h_project_array)

        download_img(h_project_img, crop_path,"05-crop_v_" + str(v_img_index) + "_h_project_img")

        h_project_array_valid = h_project_array[h_project_array > 0]
        h_project_array_max = np.max(h_project_array_valid)
        h_project_array_sum = np.sum(h_project_array_valid)
        h_project_array_median = np.median(h_project_array_valid)
        h_project_array_mean = h_project_array_sum / len(h_project_array_valid)

        print("[Message] : h_project_array_median :", h_project_array_median)
        print("[Message] : h_project_array_mean :", h_project_array_mean)
        scale = 0.1
        h_project_array_threshold = min(h_project_array_mean, h_project_array_median) * scale
        print("[Message] : h_project_array_threshold :", h_project_array_threshold)

        h_project_array[h_project_array < (h_project_array_threshold)] = 0
        print("[Message] : h_project_array :", h_project_array)
        #print("sum of h_project_array", np.sum(h_project_array))

        h_projection_img = np.zeros(cropImg.shape, dtype=np.uint8)
        cropImg_h,cropImg_w = cropImg.shape
        for i in range(len(h_project_array)):
            #if h_project_array[i] != 0:
            cv2.line(h_projection_img, (cropImg_w - h_project_array[i],i), (cropImg_w, i), (255))

        download_img(h_projection_img, crop_path, "05-crop_v_" + str(v_img_index) + "_h_project_img_done")
        #continue


        h_start_sign = 0
        h_start, h_end = [], []
        for i in range(len(h_project_array)):
            if h_project_array[i] > 0 and h_start_sign == 0:
                h_start.append(i)
                h_start_sign = 1
            if h_project_array[i] == 0 and h_start_sign == 1:
                h_end.append(i)
                h_start_sign = 0

        for h_img_index in range(len(h_start)):
            bin_height = abs(h_end[h_img_index] - h_start[h_img_index])
            #if bin_height < cropImg_w /10:
                #continue

            # 当确认了起点和终点之后保存坐标1`
            #if hend == 1:
            #    position.append([h_start, v_start[i], h_end, v_end[i]])
            #    hend = 0

            cropWord = cropImg[h_start[h_img_index]:h_end[h_img_index],0:w]

            download_img(cropWord, crop_path, "06-crop_word" + str(h_img_index))









def pre_process(img,crop_path,handwriting_is_black = True):
    # 00.高斯滤波去噪
    img = cv2.GaussianBlur(img, (5, 5), 0)

    download_img(img, crop_path,"00-GaussianBlur")

    img_gray = img
    # 01.彩色图灰度图，转灰度图

    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape  # 获取图片宽高

    download_img(img_gray, crop_path,"01-gray")

    #threshold = compute_threshold(img_gray)
    threshold = 101
    print("[Message] compute_threshold = ",threshold)
    #print(type(img_gray))


    img_gray_th = img_gray.copy()
    if  handwriting_is_black == True:
        cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV,dst=img_gray_th)
    else:
        cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY, dst=img_gray_th)
    download_img(img_gray_th, crop_path,"02-threshold")

    cv2.medianBlur(img_gray_th, 5, img_gray_th)
    download_img(img_gray_th, crop_path, "02-medianBlur")
    #开运算去噪点
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #img_gray_th = cv2.morphologyEx(img_gray_th,cv2.MORPH_OPEN,element)
    #download_img(img_gray_th, crop_path,"02")

    return img_gray_th





path = "./全部合集"
path = "./全部合集/法帖"
path = "./全部合集/碑刻"
path = "./全部合集/墨迹/中字/中中/昆陽城賦"
path = "./input"
crop_out_path  = "./output"
imgPaths = []
get_file(path, imgPaths)



for imgPath in imgPaths:
    print("#####################################")
    all_img_crop_path = crop_out_path  + "/"+ imgPath
    all_img_crop_name = os.path.splitext(all_img_crop_path)[0]

    if True == os.path.exists(crop_out_path):
        remove_dirs(crop_out_path)

    if False == os.path.exists(imgPath):
        print("[Waring] no exits :", imgPath)
        continue
    print("[Doing] ",imgPath)
    img = cv_imread(imgPath)
    if img is None:
        print("[Waring] img is empty :", imgPath)
        continue
    pre_process_img = pre_process(img,all_img_crop_name,False)
    crop_word(pre_process_img,all_img_crop_name,False)









