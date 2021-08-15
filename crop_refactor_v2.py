import logging

import cv2
import sys
import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from util import *




class CROP_WORD:
    def __init__(self,output_middle_path,output_path):
        self.output_middle_path = output_middle_path
        self.output_path = output_path
        self.clear()
        return

    def clear(self):
        self.y_project_array = 0
        self.y_project_array_threshold = 0
        self.x_project_array = []
        self.x_project_array_threshold = []
        return
    def moving_average(self,x, w):
        tmp = np.convolve(x, np.ones(w), 'valid') / w
        return np.concatenate((np.zeros(int(w / 2)), tmp))
    def moving_medlian(self,x, w):
        return scipy.signal.medfilt(x, w)
        # http://www.c-s-a.org.cn/html/2021/2/7819.html

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
        download_img(img_colour, self.output_middle_path, "00-GaussianBlur")

        img_gray = img_colour
        if img_colour.ndim == 3:
            img_gray = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
        download_img(img_gray, self.output_middle_path, "01-gray")
        height, width = img_gray.shape  # 获取图片宽高

        threshold = self.compute_threshold(img_gray)
        #threshold = 101

        logging.info("[Message] compute_threshold = %s", threshold)

        img_gray_th = img_gray.copy()
        if handwriting_is_black == True:
            cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV, dst=img_gray_th)
        else:
            cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY, dst=img_gray_th)
        download_img(img_gray_th, self.output_middle_path, "02-threshold")

        cv2.medianBlur(img_gray_th, 5, img_gray_th)
        download_img(img_gray_th, self.output_middle_path, "02-medianBlur")
        # 开运算去噪点
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        #img_gray_th = cv2.morphologyEx(img_gray_th,cv2.MORPH_OPEN,element)
        kernel = np.ones((3, 3), np.uint8)
        img_gray_th = cv2.erode(img_gray_th, kernel, iterations=1)
        download_img(img_gray_th, self.output_middle_path,"02-morphologyEx")

        return img_gray_th


    # 垂直向投影
    def YProject(self,binary):
        h, w = binary.shape

        y_projection_array = np.zeros(w).astype(np.uint32)
        for i in range(w):
            for j in range(h):
                if binary[j, i] == 255:
                    y_projection_array[i] += 1

        #logging.info("[Message] : y_projection_array :", y_projection_array)

        projectArrary_valid = y_projection_array[y_projection_array > 0]
        projectArrary_max = np.max(projectArrary_valid)
        projectArrary_sum = np.sum(projectArrary_valid)
        projectArrary_median = np.median(projectArrary_valid)
        projectArrary_mean = projectArrary_sum / len(projectArrary_valid)
        projectArrary_threshold = min(projectArrary_median, projectArrary_mean)
        y_projection_array_threshold = projectArrary_threshold
        print("[Message] :         y_projection_array = :", y_projection_array)
        print("[Message] :    max(y_projection_array) = :", projectArrary_max)
        print("[Message] :   mean(y_projection_array) = :", projectArrary_mean)
        print("[Message] : median(y_projection_array) = :", projectArrary_median)

        y_projection_img = np.zeros((binary.shape[0],binary.shape[1],3) , dtype=np.uint8)
        for i in range(len(y_projection_array)):
            if y_projection_array[i] != 0:
                cv2.line(y_projection_img, (i, h - y_projection_array[i]), (i, h), (0,255,0))

        return np.array(y_projection_array), np.array(y_projection_img),y_projection_array_threshold
    # 水平方向投影
    def XProject(self,binary):
        h, w = binary.shape

        x_projection_array = np.zeros(h).astype(np.uint32)
        for j in range(h):
            for i in range(w):
                if binary[j, i] == 255:
                    x_projection_array[j] += 1

        #logging.info("[Message] : x_projection_array :", x_projection_array)
        projectArrary_valid = x_projection_array[x_projection_array > 0]
        projectArrary_max = np.max(projectArrary_valid)
        projectArrary_sum = np.sum(projectArrary_valid)
        projectArrary_median = np.median(projectArrary_valid)
        projectArrary_mean = projectArrary_sum / len(projectArrary_valid)
        #projectArrary_threshold = max(projectArrary_median, projectArrary_mean)
        projectArrary_threshold = projectArrary_max /3 *2
        x_projection_array_threshold =projectArrary_threshold
        print("[Message] :         x_projection_array = :", x_projection_array)
        print("[Message] :    max(x_projection_array) = :", projectArrary_max)
        print("[Message] :   mean(x_projection_array) = :", projectArrary_mean)
        print("[Message] : median(x_projection_array) = :", projectArrary_median)

        x_projection_img = np.zeros((binary.shape[0],binary.shape[1],3) , dtype=np.uint8)
        for i in range(len(x_projection_array)):
            if x_projection_array[i] != 0:
                cv2.line(x_projection_img, (w - x_projection_array[i], i), (w, i), (255,0,0))
        return np.array(x_projection_array), np.array(x_projection_img),x_projection_array_threshold
    def binsPostProcess(self,bins,bin_threshold):
        #处理小峰值
        for i in range(len(bins)):
            bin_start, bin_end = bins[i]
            if (bin_end - bin_start) <= bin_threshold:
                if i == 0 and i != len(bins) -1:  #前边没有峰,后边有峰
                    bin_next_start, bin_next_end = bins[i+1]
                    bins[i+1] = (bin_start,bin_next_end)
                if i != 0 and i == len(bins) - 1:  # 前边有峰,后边没有峰
                    bin_befre_start, bin_befre_end = bins[i-1]
                    bins[i-1] = (bin_befre_start,bin_end)
                if i != 0 and i != len(bins) - 1:  # 前后都有峰
                    bin_befre_start, bin_befre_end = bins[i-1]
                    bin_next_start, bin_next_end = bins[i + 1]
                    bin_meddle = int(bin_start + (bin_end - bin_start) / 2)

                    dis_befre = bin_meddle - bin_befre_end
                    dis_next = bin_next_start - bin_meddle
                    if dis_befre > dis_next :
                        bins[i + 1] = (bin_start, bin_next_end)
                    elif dis_befre < dis_next:
                        bins[i - 1] = (bin_befre_start, bin_end)
                    elif dis_befre ==  dis_next:
                        bins[i - 1] = (bin_befre_start, bin_meddle)
                        bins[i + 1] = (bin_meddle, bin_next_end)
                bins[i] = (0,0)

        #处理被清空的峰值
        bins_result = []
        for bin_start, bin_end  in bins:
            if bin_start != 0 and bin_end != 0:
                bins_result.append((bin_start, bin_end))

        #处理距离过大的峰
        for i in range(len(bins_result)):
            bin_start, bin_end = bins_result[i]
            if i != 0 :  #只需要前边有峰

                bin_befre_start, bin_befre_end = bins_result[i - 1]

                # 当前的峰跟之前的峰,要是距离过大,则不需要处理.峰之间的缝隙,比字还大.就是隔开了
                if bin_start - bin_befre_end > self.binsAve(bins_result) :
                    continue

                through_meddle = int((bin_befre_end + bin_start ) /2)
                bins_result[i - 1] = (bin_befre_start, through_meddle)
                bins_result[i] = (through_meddle, bin_end)


        return bins_result

    def binsAve(self,bins):
        sum = 0
        index=0
        for bin_start ,bin_end in bins:
            sum = sum + (bin_end - bin_start)
            index=index+1
        if sum == 0:
            return 0
        else :
            return sum/index

    def YProjectPostProcess(self, project_array, h_project_array_threshold, bin_threshold=20):

        array_sign = np.zeros(project_array.shape)
        project_array[project_array < (h_project_array_threshold)] = 0

        bins = []
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
                if (bin_end > bin_start ):
                    bins.append((bin_start,bin_end))

        #bins 后处理器
        bins =  self.binsPostProcess(bins,bin_threshold)

        return bins

    def XProjectPostProcess(self, project_array, v_project_array_threshold, bin_threshold=20):

        array_sign = np.zeros(project_array.shape)
        project_array[project_array < (v_project_array_threshold)] = 0

        bins = []
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
                if (bin_end > bin_start ):
                    bins.append((bin_start,bin_end))

        #bins 后处理器
        bins =  self.binsPostProcess(bins,bin_threshold)

        return bins


    def drawLineFromYProjectBinsToImg(self, img_colours, bins):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_tmp = img_colours.copy()
        index = 0
        for trough_index_start,trough_index_end in bins:
            cv2.line(img_tmp, (trough_index_start, 0), (trough_index_start, img_gray_h),   (255,255,255),3)
            cv2.line(img_tmp, (trough_index_end, 0), (trough_index_end, img_gray_h),       (255,255,255),3)
            cv2.line(img_tmp, (trough_index_start, int(img_gray_h/4)), (trough_index_end, int(img_gray_h/4)), (255, 0, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            img_tmp = cv2.putText(img_tmp, str(index), (trough_index_start, int(img_gray_h/4)), font, 10, (0, 0, 255), 5)
            index=index+1
        #cv2.imshow("lines",img_gray_tmp)
        #cv2.waitKey(0)
        return img_tmp

    def drawLineFromXProjectBinsToImg(self, img_colours, trough_index_start,trough_index_end,bins):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_tmp = img_colours.copy()
        index = 0
        cv2.line(img_tmp, (trough_index_start, 0), (trough_index_start, img_gray_h),   (255,255,255),3)
        cv2.line(img_tmp, (trough_index_end, 0), (trough_index_end, img_gray_h),       (255,255,255),3)

        for y_index_start,y_index_end in bins:
            cv2.line(img_tmp, (trough_index_start, y_index_start), (trough_index_end, y_index_start),   (255,255,255),     3)
            cv2.line(img_tmp, (trough_index_start, y_index_end),   (trough_index_end, y_index_end),   (255, 255, 255),   3)
        return img_tmp



    def find_word_box(self,img_gray,img_colours_src):

        self.clear()
        img_colours = img_colours_src.copy()
        h, w = img_gray.shape
        y_project_array, y_project_img,y_projection_array_threshold = self.YProject(img_gray)

        self.y_project_array = y_project_array
        # 此处需要计算或者调节阈值
        y_project_array_threshold = y_projection_array_threshold * 0.25
        self.y_project_array_threshold = y_project_array_threshold
        #YProject_troughs_bins = self.YProjectPostProcess(y_project_array,self.y_project_array_threshold,self.y_project_array_threshold)
        YProject_troughs_bins = self.YProjectPostProcess(y_project_array, self.y_project_array_threshold)
        #img_colours_draw = self.drawLineFromArraryToImg(img_colours,YProject_extreme_point_array)
        img_colours = self.drawLineFromYProjectBinsToImg(img_colours,YProject_troughs_bins)
        download_img(img_colours, self.output_middle_path, "05-drawLineFromYProjectBinsToImg")

        #根据后处理出来的bins,在原图上一列一列的显示出来
        img_gray_h, img_gray_w = img_gray.shape


        index = 0
        index_crop_word = 0
        for trough_index_start,trough_index_end in YProject_troughs_bins:
            if trough_index_start < 0 or trough_index_start > img_gray_w or  trough_index_end < 0 or trough_index_end > img_gray_w or trough_index_start >= trough_index_end:
                logging.warning(["[Waring] YProject_troughs_bin[",trough_index_start,",",trough_index_end,"] continue !!"])
                continue


            #一列一列的图像,debug用
            img_gray_copy = img_gray.copy()
            #img_gray_troughs_bin = img_gray_copy[:,trough_index_start:trough_index_end]
            #download_img(img_gray_troughs_bin, self.output_path, "05-img_gray_YProject_troughs_bin_" + str(index))


            img_gray_mask = img_gray.copy()
            img_gray_mask[:,:]= 0
            img_gray_mask[:, trough_index_start:trough_index_end] = 1
            #download_img(img_gray_mask, self.output_path, "06-img_gray_YProject_troughs_bin_" + str(index))

            #在原图中只显示当前列
            img_gray_troughs_bin = img_gray_copy *img_gray_mask
            #download_img(img_gray_troughs_bin, self.output_middle_path, "06-img_gray_YProject_troughs_bin_" + str(index))

            #提取出来的当前列图像需要滤波处理
            cv2.medianBlur(img_gray_troughs_bin, 7, img_gray_troughs_bin)
            x_project_array, x_project_img,x_projection_array_threshold  = self.XProject(img_gray_troughs_bin)

            self.x_project_array.append(x_project_array)
            # 此处需要计算或者调节阈值
            x_project_array_threshold = x_projection_array_threshold * 0
            #x_project_array_threshold = 3
            self.x_project_array_threshold.append(x_project_array_threshold)
            XProject_troughs_bins = self.XProjectPostProcess(x_project_array, self.x_project_array_threshold[index],x_projection_array_threshold)

            # img_colours_draw = self.drawLineFromArraryToImg(img_colours,YProject_extreme_point_array)
            img_colours = self.drawLineFromXProjectBinsToImg(img_colours, trough_index_start,trough_index_end,XProject_troughs_bins)


            for i in range(len(XProject_troughs_bins)):
                y_index_start, y_index_end = XProject_troughs_bins[i]
                img_color_crop_word = img_colours_src[y_index_start:y_index_end,trough_index_start:trough_index_end]
                download_img(img_color_crop_word, self.output_path, "_" + str(index_crop_word))
                index_crop_word=index_crop_word+1

            #必须在循环的最后
            index = index + 1
        download_img(img_colours, self.output_middle_path, "07-drawLineFromXProjectBinsToImg")



if __name__ == '__main__':
    #print(cv2.__version__)
    input_path  = "./input"
    output_middle_path = "./output_middle"
    output_path = "./output"
    if True == os.path.exists(output_path):
        remove_dirs(output_path)
    if True == os.path.exists(output_middle_path):
        remove_dirs(output_middle_path)

    img_paths = []
    get_file(input_path, img_paths)


    for img_path in img_paths:
        logging.info("##############################################")
        crop_out_path = output_path + "/" + img_path
        crop_out_name = os.path.splitext(crop_out_path)[0]

        crop_out_middle_path = output_middle_path + "/" + img_path
        crop_out_middle_name = os.path.splitext(crop_out_middle_path)[0]

        worker = CROP_WORD(crop_out_middle_name,crop_out_name)

        if False == os.path.exists(img_path):
            logging.warning("[Waring] file no exits : %s", img_path)
            continue

        logging.info("[Doing] %s",img_path)
        img = cv_imread(img_path)
        if img is None:
            logging.warning("[Waring] img is empty :%s", img_path)
            continue
        input = img
        input = worker.pre_process(input)
        worker.find_word_box(input,img)

