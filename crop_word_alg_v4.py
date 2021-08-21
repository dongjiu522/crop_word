import cv2
import sys
import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from util import *

class CROP_WORD_ALG:
    def __init__(self):
        return
    def moving_average(self,x, w):
        tmp = np.convolve(x, np.ones(w), 'valid') / w
        return np.concatenate((np.zeros(int(w / 2)), tmp))

    def moving_medlian(self,x, w):
        return scipy.signal.medfilt(x, w)

    def setOutputMiddlePath(self,path):
        self.output_middle_path = path

    def compute_threshold(self, img_gray):
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
        download_img(img_colour, self.output_middle_path, "01-GaussianBlur")

        img_gray = img_colour
        if img_colour.ndim == 3:
            img_gray = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
        download_img(img_gray, self.output_middle_path, "02-gray")
        height, width = img_gray.shape  # 获取图片宽高

        threshold = self.compute_threshold(img_gray)
        #threshold = 101

        logging.info("[Message] compute_threshold = %s", threshold)

        img_gray_th = img_gray.copy()
        if handwriting_is_black == True:
            cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV, dst=img_gray_th)
        else:
            cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY, dst=img_gray_th)
        download_img(img_gray_th, self.output_middle_path, "03-threshold")

        cv2.medianBlur(img_gray_th, 5, img_gray_th)
        download_img(img_gray_th, self.output_middle_path, "04-medianBlur")
        # 开运算去噪点
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        #img_gray_th = cv2.morphologyEx(img_gray_th,cv2.MORPH_OPEN,element)
        kernel = np.ones((3, 3), np.uint8)
        img_gray_th = cv2.erode(img_gray_th, kernel, iterations=1)
        download_img(img_gray_th, self.output_middle_path,"05-erode")

        return img_gray_th

    # 垂直向投影
    def YProject(self,binary):
        return np.sum(binary, axis=0) /255
    # 水平方向投影
    def XProject(self,binary):
        return np.sum(binary, axis=1) /255
