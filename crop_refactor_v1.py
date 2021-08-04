import cv2
import sys
import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from util import *

class CROP_WORD:
    def __init__(self,output_path):
        self.output_path = output_path
        return
    # http://www.c-s-a.org.cn/html/2021/2/7819.html
    def compute_threshold(self,img_gray):
        img_max = np.max(img_gray)
        img_min = np.min(img_gray)
        img_th = img_gray.copy()
        max_g = 0
        max_g_th = 0
        for th in range(img_min, img_max):
            pixel_size = img_th.size
            more_th_size = np.sum(img_gray > th)
            low_th_size = np.sum(img_gray <= th)

            m0 = more_th_size / pixel_size
            n0 = np.mean(img_gray[img_gray > th])

            m1 = low_th_size / pixel_size
            n1 = np.mean(img_gray[img_gray <= th])

            g = m0 * m1 * ((n0 - n1) ** 2)
            if g > max_g:
                max_g = g
                max_g_th = th
        return max_g_th

    def pre_process(self, img_colour, handwriting_is_black=False):


        # 00.高斯滤波去噪
        img_colour = cv2.GaussianBlur(img_colour, (5, 5), 0)
        download_img(img_colour, self.output_path, "00-GaussianBlur")

        img_gray = img_colour
        if img_colour.ndim == 3:
            img_gray = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
        download_img(img_gray, self.output_path, "01-gray")
        height, width = img_gray.shape  # 获取图片宽高

        #threshold = self.compute_threshold(img_gray)
        threshold = 101

        print("[Message] compute_threshold = ", threshold)
        img_gray_th = img_gray.copy()
        if handwriting_is_black == True:
            cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV, dst=img_gray_th)
        else:
            cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY, dst=img_gray_th)
        download_img(img_gray_th, self.output_path, "02-threshold")

        cv2.medianBlur(img_gray_th, 5, img_gray_th)
        download_img(img_gray_th, self.output_path, "02-medianBlur")
        # 开运算去噪点
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # img_gray_th = cv2.morphologyEx(img_gray_th,cv2.MORPH_OPEN,element)
        # download_img(img_gray_th, crop_path,"02")

        return img_gray_th

    # 水平方向投影
    def XProject(self,binary):
        h, w = binary.shape

        x_projection_array = [0] * h
        for j in range(h):
            for i in range(w):
                if binary[j, i] == 255:
                    x_projection_array[j] += 1

        print("[Message] : x_projection_array :", x_projection_array)

        x_projection_img = np.zeros((binary.shape[0],binary.shape[1],3) , dtype=np.uint8)
        for i in range(len(x_projection_array)):
            if x_projection_array[i] != 0:
                cv2.line(x_projection_img, (w - x_projection_array[i], i), (w, i), (255,0,0))
        return np.array(x_projection_array), np.array(x_projection_img)

    # 垂直向投影
    def YProject(self,binary):
        h, w = binary.shape

        y_projection_array = [0] * w
        for i in range(w):
            for j in range(h):
                if binary[j, i] == 255:
                    y_projection_array[i] += 1
        print("[Message] : y_projection_array :", y_projection_array)

        y_projection_img = np.zeros((binary.shape[0],binary.shape[1],3) , dtype=np.uint8)
        for i in range(len(y_projection_array)):
            if y_projection_array[i] != 0:
                cv2.line(y_projection_img, (i, h - y_projection_array[i]), (i, h), (0,255,0))

        return np.array(y_projection_array), np.array(y_projection_img)

    def find_word_box(self,img_gray):
        h, w = img_gray.shape
        y_project_array, y_project_img = self.YProject(img_gray)

        #y_project_array = signal.medfilt(y_project_array, 9)

        #signal.find_peaks(y_project_array)

        b, a = signal.butter(8, 0.8, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, y_project_array)  # data为要过滤的信号
        self.signal_show(filtedData)

    def signal_show(self,signal_val):
        signal_val = np.array(signal_val)

        xxx = range(len(signal_val))
        yyy = signal_val

        z1 = np.polyfit(xxx, yyy, 25)  # 用7次多项式拟合
        p1 = np.poly1d(z1)  # 多项式系数

        yvals = p1(xxx)


        num_peak = signal.find_peaks(yvals, distance=10)  # distance表极大值点的距离至少大于等于10个水平单位
        print(num_peak[0])
        print('the number of peaks is ' + str(len(num_peak[0])))

        #plt.plot(xxx, yyy, '*', label='original values')
        plt.plot(xxx, yvals, 's', label='signal values')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend(loc=4)
        plt.title('signal')
        for ii in range(len(num_peak[0])):
            plt.plot(num_peak[0][ii], signal_val[num_peak[0][ii]], '*', markersize=10)
        plt.show()






if __name__ == '__main__':
    #print(cv2.__version__)
    input_path  = "./input"
    output_path = "./output"

    if True == os.path.exists(output_path):
        remove_dirs(output_path)

    img_paths = []
    get_file(input_path, img_paths)


    for img_path in img_paths:
        print("#####################################")
        crop_out_path = output_path + "/" + img_path
        crop_out_name = os.path.splitext(crop_out_path)[0]

        worker = CROP_WORD(crop_out_name)

        if False == os.path.exists(img_path):
            print("[Waring] file no exits :", img_path)
            continue

        print("[Doing] ", img_path)
        img = cv_imread(img_path)
        if img is None:
            print("[Waring] img is empty :", img_path)
            continue
        input = img
        input = worker.pre_process(input)
        worker.find_word_box(input)

