import cv2
import math
import sys
import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from util import *

class CROP_WORD_ALG:
    output_middle_path = "./output_middle"

    is_handwriting_black = False
    is_auto_threshold    = True

    img_gray_threshold = 0


    y_projection_img = []
    y_project_threshold= 0
    y_project_threshold_scale = 0.5
    y_projection_array_th_img = []

    y_bins = []
    y_bin_width_threshold = 20
    y_bins_dis_threshold = 5

    def setOutputMiddlePath(self,path):
        self.output_middle_path = path

    def setHandWritingIsBlack(self,flage):
        self.is_handwriting_black = flage

    def getImgGrayThreshold(self):
        return self.img_gray_threshold

    def setIsAutoThreshold(self,is_auto_threshold, threshold = 0):
        self.is_auto_threshold = is_auto_threshold
        if is_auto_threshold == False :
            self.img_gray_threshold = threshold
            logging.info("[Waring] change img_gray_threshold  = %s", self.img_gray_threshold)

    def getYprojectionImg(self):
        return self.y_projection_img

    def getYProjectThreshold(self):
        return self.y_project_threshold

    def setYProjectThreshold(self,t):
        self.y_project_threshold = t

    def getYProjectThresholdScale(self):
        return self.y_project_threshold_scale

    def setYProjectThresholdScale(self, scale):
        self.y_project_threshold_scale = scale

    def getYprojectionImgTh(self):
        return self.y_projection_array_th_img

    def getYprojectionBins(self):
        return self.y_bins

    def getYprojectionBinWidthThreshold(self):
        return self.y_bin_width_threshold
    def setYprojectionBinWidthThreshold(self,t):
        self.y_bin_width_threshold = t
    def getYprojectionBinDisThreshold(self):
        return self.y_bins_dis_threshold
    def setYprojectionBinDisThreshold(self,t):
        self.y_bins_dis_threshold = t




    #通用函数
    def moving_average(self,x, w):
        tmp = np.convolve(x, np.ones(w), 'valid') / w
        return np.concatenate((np.zeros(math.ceil(w / 2)), tmp))

    def moving_medlian(self,x, w):
        return scipy.signal.medfilt(x, w)

    def arrayThreshold(self,array,th):
        tmp = array.copy()
        tmp[tmp < (th)] = 0
        return tmp



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

    def pre_process(self,input ,is_auto_threshold,is_handwriting_black):

        # 00.高斯滤波去噪
        input = cv2.GaussianBlur(input, (5, 5), 0)
        download_img(input, self.output_middle_path, "01-GaussianBlur")

        img_gray = input
        if input.ndim == 3:
            img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        download_img(img_gray, self.output_middle_path, "02-gray")
        height, width = img_gray.shape  # 获取图片宽高

        if is_auto_threshold == True :
            self.img_gray_threshold = self.compute_threshold(img_gray)


        logging.info("[Message] compute_threshold = %s", self.img_gray_threshold)

        img_gray_th = img_gray.copy()
        if is_handwriting_black == True:
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



    def computeYProjectThreshold(self,y_projection_array):

        if np.max(y_projection_array) == 0:
            return 0
        projectArrary_valid = y_projection_array[y_projection_array > 0]
        projectArrary_max = np.max(projectArrary_valid)
        projectArrary_sum = np.sum(projectArrary_valid)
        projectArrary_median = np.median(projectArrary_valid)
        projectArrary_mean = 0
        if len(projectArrary_valid) != 0:
            projectArrary_mean = projectArrary_sum / len(projectArrary_valid)
        projectArrary_threshold = min(projectArrary_median, projectArrary_mean)

        return projectArrary_threshold

    # 垂直向投影
    def YProject(self,binary):
        y_projection_array = np.sum(binary, axis=0) /255
        y_projection_array = self.moving_average(y_projection_array,5)
        return  np.array(y_projection_array)

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

    def YProjectPostProcessToBins(self, project_array):
        project_array = np.array(project_array)
        array_sign = np.zeros(project_array.shape)
        #project_array[project_array < (h_project_array_threshold)] = 0

        y_bins = []
        bins_ave = 0
        bin_start = 0
        bin_end = 0
        bin_flage = 0
        for i in range(len(project_array)):
            if project_array[i] != 0 and bin_flage == 0:
                bin_start = i
                bin_flage = 1
            if (project_array[i] == 0 or i >= len(project_array) - 1) and bin_flage == 1:
                bin_end = i
                bin_flage = 0
                if (bin_end > bin_start):
                    y_bins.append((bin_start, bin_end,1))

        return y_bins

    def binsPostProcess(self,project_array,bins,bin_width_threshold,bins_dis_threshold):

        #先处理两个离得近的峰

        #再处理小峰
        for i in range(len(bins)):
            bin_start, bin_end,flag = bins[i]
            if flag != 1:
                continue
            if (bin_end - bin_start) <= bin_width_threshold:
                if i == 0 and i != len(bins) -1:  #前边没有峰,后边有峰
                    bin_next_start, bin_next_end,flag = bins[i+1]
                    bins[i+1] = (bin_start,bin_next_end,1)
                if i != 0 and i == len(bins) - 1:  # 前边有峰,后边没有峰
                    bin_befre_start, bin_befre_en,flag = bins[i-1]
                    bins[i-1] = (bin_befre_start,bin_end,1)
                if i != 0 and i != len(bins) - 1:  # 前后都有峰
                    bin_befre_start, bin_befre_end,flag = bins[i-1]
                    bin_next_start, bin_next_end,flag = bins[i + 1]
                    bin_meddle = int(bin_start + (bin_end - bin_start) / 2)
                    #寻找这个小峰的数值最大的线为峰线,通过这个峰线来判断距离左右两边那边最近,然后融合
                    project_array_bin_crop = project_array[bin_start:bin_end]
                    project_array_bin_crop_max = np.max(project_array_bin_crop)
                    project_array_bin_crop_max_indexs = np.where(project_array_bin_crop_max)
                    #bin_meddle = bin_start + np.mean(project_array_bin_crop_max_indexs)

                    dis_befre = bin_meddle - bin_befre_end
                    dis_next = bin_next_start - bin_meddle
                    if dis_befre > dis_next :
                        bins[i + 1] = (bin_start, bin_next_end,1)
                    elif dis_befre < dis_next:
                        bins[i - 1] = (bin_befre_start, bin_end,1)
                    elif dis_befre ==  dis_next:
                        bins[i - 1] = (bin_befre_start, bin_meddle,1)
                        bins[i + 1] = (bin_meddle, bin_next_end,1)
                bins[i] = (0,0,0)


        #处理被清空的峰值
        bins_result = []
        for bin_start, bin_end,flag  in bins:
            if flag != 1:
                continue
            #if bin_start != 0 and bin_end != 0:
            bins_result.append((bin_start, bin_end,1))

        #处理距离过大的峰
        for i in range(len(bins_result)):
            bin_start, bin_end,flag = bins_result[i]
            if flag != 1:
                continue
            if i != 0 :  #只需要前边有峰

                bin_befre_start, bin_befre_end,flag = bins_result[i - 1]

                # 当前的峰跟之前的峰,要是距离过大,则不需要处理.峰之间的缝隙,比字还大.就是隔开了
                if bin_start - bin_befre_end > bins_dis_threshold :
                    continue
                #此处寻找这个峰被两变的峰融合掉了,找中间分割点最小之值作为分割线
                through_meddle = int((bin_befre_end + bin_start) / 2)
                project_array_bin_crop = project_array[bin_befre_end:bin_start]
                project_array_bin_crop_max = np.min(project_array_bin_crop)
                project_array_bin_crop_max_indexs = np.where(project_array_bin_crop_max)
                #through_meddle =  bin_start + np.mean(project_array_bin_crop_max_indexs)

                bins_result[i - 1] = (bin_befre_start, through_meddle)
                bins_result[i] = (through_meddle, bin_end)


        #此处需要扩展两边缘的bin的宽度,让其能够将两边的字的边缘能够包含进去

        return bins_result

    def drawLineFromYProjectBinsToImg(self, img_colours, bins):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_tmp = img_colours.copy()
        index = 0
        for trough_index_start,trough_index_end,flag in bins:
            if flag != 1:
                continue
            cv2.line(img_tmp, (trough_index_start, 0), (trough_index_start, img_gray_h),   (255,0,0),2)
            cv2.line(img_tmp, (trough_index_end, 0), (trough_index_end, img_gray_h),       (0,0,255),2)
            cv2.line(img_tmp, (trough_index_start, int(img_gray_h/4)), (trough_index_end, int(img_gray_h/4)), (0, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            #img_tmp = cv2.putText(img_tmp, str(index), (trough_index_start, int(img_gray_h/4)), font, 10, (0, 0, 0), 1)
            index=index+1
        return img_tmp

    def drawLineFromYProjectThresholdToImg(self, img_colours, threshold):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_tmp = img_colours.copy()
        threshold = int(min(threshold,img_gray_h))
        cv2.line(img_tmp, (0, img_gray_h - threshold), (img_gray_w, img_gray_h - threshold), (0, 0, 0), 3)
        return img_tmp


    # 水平方向投影
    def XProject(self,binary):
        x_projection_array = np.sum(binary, axis=1) /255
        x_projection_array = self.moving_average(x_projection_array,5)
        return x_projection_array

    def converToBinsFromSignPeakArray(self,array):
        peaks_satrt = np.where(array == -1)
        peaks_end   = np.where(array == -10)
        peaks_w = peaks_end - peaks_satrt
        peaks_w[peaks_w <= 0] = 0
        return np.average(peaks_w)

    def auto_work(self,input):
        #预处理
        self.img_gray_pre_processed = self.pre_process(input,self.is_auto_threshold,self.is_handwriting_black)

        #垂直投影
        self.y_projection_array =  self.YProject(self.img_gray_pre_processed)

        #垂直投影转img
        self.y_projection_img = self.converYProjectToImg(self.y_projection_array)
        download_img(self.y_projection_img, self.output_middle_path, "06-y_projection_img")

        #计算下垂直投影的建议阈值,并乘以缩放系数
        self.y_project_threshold = self.computeYProjectThreshold(self.y_projection_array)
        self.y_project_threshold= int(self.y_project_threshold * self.y_project_threshold_scale)

        #根据上一步的阈值将垂直投影阈值化
        self.y_projection_array_th = self.arrayThreshold(self.y_projection_array,self.y_project_threshold)

        #将阈值化后的垂直投影转换成img
        self.y_projection_array_th_img = self.converYProjectToImg(self.y_projection_array_th)
        download_img(self.y_projection_array_th_img, self.output_middle_path, "07-y_projection_array_th_img")

        #将垂直投影转成bins
        self.y_bins = self.YProjectPostProcessToBins(self.y_projection_array_th)

        #计算bins后处理的宽度阈值和距离阈值
        #应该不用计算

        #垂直投影bins后处理
        self.y_bins = self.binsPostProcess(self.y_projection_array_th,self.y_bins,self.y_bin_width_threshold,self.y_bins_dis_threshold)

        #将垂直投影的bins画到图像上
        self.y_projection_img = self.drawLineFromYProjectBinsToImg(self.y_projection_img,self.y_bins)
        self.y_projection_img = self.drawLineFromYProjectThresholdToImg(self.y_projection_img,self.y_project_threshold)
        self.y_projection_array_th_img = self.drawLineFromYProjectBinsToImg(self.y_projection_array_th_img, self.y_bins)



        return


