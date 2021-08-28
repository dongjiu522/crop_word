import cv2
import math
import sys
import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from util import *

class CROP_WORD_ALG:
    is_handwriting_black = False
    is_auto_threshold    = True
    img_gray_threshold = 0
    y_project_threshold= 0
    y_project_threshold_scale = 0.5

    def setOutputMiddlePath(self,path):
        self.output_middle_path = path

    def getYProjectThreshold(self):
        return self.y_project_threshold

    def setYProjectThreshold(self,t):
        self.y_project_threshold = t

    def setIsAutoThreshold(self,mode, threshold = 0):
        self.is_auto_threshold = mode
        if mode == False :
            self.img_gray_threshold = threshold
            logging.info("[Waring] change img_gray_threshold  = %s", self.img_gray_threshold)
    def getImgGrayThreshold(self):
        return self.img_gray_threshold

    def moving_average(self,x, w):
        tmp = np.convolve(x, np.ones(w), 'valid') / w
        return np.concatenate((np.zeros(math.ceil(w / 2)), tmp))

    def moving_medlian(self,x, w):
        return scipy.signal.medfilt(x, w)

    def arrayThreshold(self,array,th):
        tmp = array.copy()
        tmp[tmp < (th)] = 0
        return tmp


    def setHandWritingIsBlack(self,flage):
        self.is_handwriting_black = flage

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

    def pre_process(self,input ):

        # 00.高斯滤波去噪
        input = cv2.GaussianBlur(input, (5, 5), 0)
        download_img(input, self.output_middle_path, "01-GaussianBlur")

        img_gray = input
        if input.ndim == 3:
            img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        download_img(img_gray, self.output_middle_path, "02-gray")
        height, width = img_gray.shape  # 获取图片宽高

        if self.is_auto_threshold == True :
            self.img_gray_threshold = self.compute_threshold(img_gray)


        logging.info("[Message] compute_threshold = %s", self.img_gray_threshold)

        img_gray_th = img_gray.copy()
        if self.is_handwriting_black == True:
            cv2.threshold(img_gray, self.img_gray_threshold, 255, cv2.THRESH_BINARY_INV, dst=img_gray_th)
        else:
            cv2.threshold(img_gray, self.img_gray_threshold, 255, cv2.THRESH_BINARY, dst=img_gray_th)
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

    def converYProjectToImg(self,y_projection_array):
        y_projection_array_max = np.max(y_projection_array).astype(int)

        #保护下,防止图像大小为0
        y_projection_array_max = max(20,y_projection_array_max)
        y_projection_img = np.zeros((int(y_projection_array_max), len(y_projection_array), 3), dtype=np.uint8)
        y_projection_img[:] = 255
        for i in range(len(y_projection_array)):
            if y_projection_array[i] != 0:
                cv2.line(y_projection_img, (i, int(y_projection_array_max - y_projection_array[i])),
                         (i, int(y_projection_array_max)), (255, 0, 0))
        return y_projection_img

    def computeYProjectThreshold(self):

        if np.max(self.y_projection_array) == 0:
            return 0
        projectArrary_valid = self.y_projection_array[self.y_projection_array > 0]
        projectArrary_max = np.max(projectArrary_valid)
        projectArrary_sum = np.sum(projectArrary_valid)
        projectArrary_median = np.median(projectArrary_valid)
        projectArrary_mean = 0
        if len(projectArrary_valid) != 0:
            projectArrary_mean = projectArrary_sum / len(projectArrary_valid)
        projectArrary_threshold = min(projectArrary_median, projectArrary_mean)

        return projectArrary_threshold * self.y_project_threshold_scale

    # 垂直向投影
    def YProject(self,binary):
        y_projection_array = np.sum(binary, axis=0) /255
        y_projection_array = self.moving_average(y_projection_array,5)
        return  y_projection_array
    def converToBinsFromSignPeakArray(self,array):
        peaks_satrt = np.where(array == -1)
        peaks_end   = np.where(array == -10)
        peaks_w = peaks_end - peaks_satrt
        peaks_w[peaks_w <= 0] = 0
        return np.average(peaks_w)
    def YProjectPostProcess(self,y_projection_array):
        #print(np.where(y_projection_array == 0))
        y_projection_array_sign_peaks =  np.zeros(y_projection_array.shap).astype(np.uint32)
        y_projection_array_sign_peaks_max_val = np.zeros(y_projection_array.shap).astype(np.uint32)
        y_projection_array_peaks_index = np.where(y_projection_array > 0)

        peak_start_index = -1
        peak_end_index   = -1
        sign = False
        for i in range(len(y_projection_array_peaks_index) -1):
            if sign == False:
                peak_start_index = y_projection_array_peaks_index[i]
                sign = True

            if sign == True and y_projection_array_peaks_index[i] != y_projection_array_peaks_index[i+1]:
                sign = False
                peak_end_index = y_projection_array_peaks_index[i]

                if peak_start_index > 0 and peak_end_index - peak_start_index > 4:
                    y_projection_array_sign_peaks[peak_start_index] = -1
                    y_projection_array_sign_peaks[peak_end_index] = -10
                    peak = y_projection_array(peak_start_index+1,peak_end_index-1)
                    peak_max_val = np.max(peak)
                    peak_max_index = np.where(peak == peak_max_val)
                    peak_max_index = np.median(peak_max_index)
                    y_projection_array_sign_peaks[peak_start_index+1 + peak_max_index] = peak_max_val


        #对于距离很近的两个峰,进行合并

        #处理左右两边的情况




        #
        self.computBinsAveWidthFromSignPeakArray()















    # 水平方向投影
    def XProject(self,binary):
        return np.sum(binary, axis=1) /255

    def auto_work(self,input):
        self.img_gray_pre_processed = self.pre_process(input)


        self.y_projection_array =  self.YProject(self.img_gray_pre_processed)
        self.y_projection_array_ave = self.moving_average(self.y_projection_array, 5)
        self.y_projection_img = self.converYProjectToImg(self.y_projection_array_ave)
        download_img(self.y_projection_img, self.output_middle_path, "06-y_projection_img")

        self.y_project_threshold = self.computeYProjectThreshold()
        self.y_projection_array_th = self.arrayThreshold(self.y_projection_array_ave,self.y_project_threshold)
        #self.y_projection_array_th = self.moving_average(self.y_projection_array_th,5)
        self.y_projection_array_th_img = self.converYProjectToImg(self.y_projection_array_th)
        download_img(self.y_projection_array_th_img, self.output_middle_path, "07-y_projection_array_th_img")

        self.YProjectPostProcess(self.y_projection_array_th)

        return self.y_projection_img,self.y_projection_array_th_img


