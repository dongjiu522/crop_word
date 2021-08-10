import logging

import cv2
import sys
import os
import numpy as np
import scipy.signal
import scipy.fft
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

        logging.info("[Message] compute_threshold = %s", threshold)

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

    def moving_average(self,x, w):
        tmp = np.convolve(x, np.ones(w), 'valid') / w
        return np.concatenate((np.zeros(int(w / 2)), tmp))
    def moving_medlian(self,x, w):
        return scipy.signal.medfilt(x, w)

    #% 功能：对一维信号的高斯滤波，头尾r / 2的信号不进行滤波
    #% src: 需要进行高斯滤波的序列
    #% sigma: 标准差
    #% r: 高斯模板的大小推荐奇数
    # https://loopvoid.github.io/2017/03/04/%E4%B8%80%E7%BB%B4%E4%BF%A1%E5%8F%B7%E7%9A%84%E9%AB%98%E6%96%AF%E6%BB%A4%E6%B3%A2/
    def gauss_filter(self,src,r,sigma = -1):


        kernel_size = r*2 +1
        print("[Message] kernel_size = ", kernel_size)
        kernel = np.zeros(kernel_size)
        center = r + 1
        if sigma <= 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

        s = sigma ** 2

        sum_val = 0
        for i in range(kernel_size):
                x = i - center
                kernel[i] = np.exp(-(x ** 2) / 2 * s)
                sum_val += kernel[i]
        kernel = kernel / sum_val

        x_filted = np.zeros(src.shape)
        for i in range(r,len(src)-r-1):
            x_filted[i] = np.sum(src[i-r : i+r+1]*kernel)*kernel[center]

        return x_filted




    # 水平方向投影
    def XProject(self,binary):
        h, w = binary.shape

        x_projection_array = [0] * h
        for j in range(h):
            for i in range(w):
                if binary[j, i] == 255:
                    x_projection_array[j] += 1

        #logging.info("[Message] : x_projection_array :", x_projection_array)

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

        #logging.info("[Message] : y_projection_array :", y_projection_array)
        print("[Message] : y_projection_array :", y_projection_array)
        y_projection_img = np.zeros((binary.shape[0],binary.shape[1],3) , dtype=np.uint8)
        for i in range(len(y_projection_array)):
            if y_projection_array[i] != 0:
                cv2.line(y_projection_img, (i, h - y_projection_array[i]), (i, h), (0,255,0))

        return np.array(y_projection_array), np.array(y_projection_img)


    def getPeaksAndTroughs(self,h, rangesize):
        h = np.array(h).astype(np.uint32)
        peaks = list()
        troughs = list()
        S = 0
        for x in range(0, len(h) -1):

            if S == 0:
                if h[x] > h[x + 1]:
                    S = 1  ## down
                else:
                    S = 2  ## up

            elif S == 1:  ### down
                if h[x] < h[x + 1]:
                    S = 2
                    ## from down to up
                    if len(troughs):
                        ## check if need merge
                        (prev_x, prev_trough) = troughs[-1]
                        if x - prev_x < rangesize:
                            if prev_trough > h[x]:
                                troughs[-1] = (x, h[x])
                        else:
                            troughs.append((x, h[x]))
                    else:
                        troughs.append((x, h[x]))


            elif S == 2:  ## up
                if h[x] > h[x + 1]:
                    S = 1
                    ## from up to down
                    if len(peaks):
                        prev_x, prev_peak = peaks[-1]
                        if x - prev_x < rangesize:
                            if prev_peak < h[x]:
                                peaks[-1] = (x, h[x])
                        else:
                            peaks.append((x, h[x]))
                    else:
                        peaks.append((x, h[x]))

            if x == 0:
                if h[x] > h[x + 1]:
                    peaks.append((x,h[x]))
                else:
                    troughs.append((x, h[x]))
            if x == len(h) -2:
                if h[x] > h[x + 1]:
                    troughs.append((x+1, h[x+1]))
                else:
                    peaks.append((x+1, h[x+1]))
        return peaks, troughs



    def convertPeaksAndTroughsToArrary(self,src_array,src_peaks,peaks_threshold,src_troughs,troughs_threshold):

        dst_array = np.zeros(src_array.shape)
        for x, y in src_peaks:
            if y >= peaks_threshold :
                dst_array[x] = 100
            else :
                dst_array[x] = -50

        for x, y in src_troughs:
            if y < troughs_threshold:
                dst_array[x] = -100
            else:
                dst_array[x] = 50
        return dst_array
    def peaksAndTroughsArraryPostProcess(self,arrary):
        start = 0

        trough_index = 0
        bins = []
        for i in range(len(arrary)):
            if arrary[i] == -100:
                bins.append((trough_index,i))
                trough_index=i
            if i == len(arrary) - 1:
                bins.append((trough_index,i))
                trough_index=i
        troughs = []
        for trough_index_start,trough_index_end in bins:
            for index in range(trough_index_start,trough_index_end):
                if index < 0  or  index > len(arrary) -1:
                    continue
                if arrary[index] == 100:
                    troughs.append((trough_index_start,trough_index_end))
                    break

        return troughs
    def drawLineFromArraryToImg(self, img_colours, array):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_gray_tmp = img_colours.copy()
        for i in range(len(array)):
            if array[i] > 0 :
                cv2.line(img_gray_tmp, (i, 0), (i, img_gray_h), (255,0,0),5)
            if array[i] < 0 :
                cv2.line(img_gray_tmp, (i, 0), (i, img_gray_h), (0,0,255),5)
        #cv2.imshow("lines",img_gray_tmp)
        #cv2.waitKey(0)
        return img_gray_tmp

    def drawLineFromTroughsToImg(self, img_colours, troughs):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_gray_tmp = img_colours.copy()
        index = 0
        for trough_index_start,trough_index_end in troughs:
            cv2.line(img_gray_tmp, (trough_index_start, 0), (trough_index_start, img_gray_h),   (255,255,255),3)
            cv2.line(img_gray_tmp, (trough_index_end, 0), (trough_index_end, img_gray_h),       (255,255,255),3)
            cv2.line(img_gray_tmp, (trough_index_start, int(img_gray_h/4)), (trough_index_end, int(img_gray_h/4)), (255, 0, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            img_gray_tmp = cv2.putText(img_gray_tmp, str(index), (trough_index_start, int(img_gray_h/4)), font, 10, (0, 0, 255), 5)
            index=index+1
        #cv2.imshow("lines",img_gray_tmp)
        #cv2.waitKey(0)
        return img_gray_tmp
    def YProjectPostProcess(self, projectArrary,scale):

        projectArrary_valid = projectArrary[projectArrary > 0]

        projectArrary_max = np.max(projectArrary_valid)
        projectArrary_sum = np.sum(projectArrary_valid)
        projectArrary_median = np.median(projectArrary_valid)
        projectArrary_mean = projectArrary_sum / len(projectArrary_valid)
        projectArrary_threshold = min(projectArrary_median, projectArrary_mean)

        logging.info("[Message] : projectArrary_median :%s", projectArrary_median)
        logging.info("[Message] : projectArrary_mean :%s", projectArrary_mean)
        logging.info("[Message] : projectArrary_threshold :%s", projectArrary_threshold)

        tmp = projectArrary.copy()
        tmp[tmp < projectArrary_median] = 0
        projectArrary_median_th_array = tmp

        start = 0
        start_index = 0
        end_index = 0
        bins = []
        #self.array_show(projectArrary_median_th_array)
        for i in range(len(projectArrary_median_th_array)):
            if projectArrary_median_th_array[i] > 0 and start == 0:
                start_index = i
                start = 1

            if ( projectArrary_median_th_array[i] <= 0 or i >= len(projectArrary_median_th_array)  )and start == 1:
                end_index = i
                start = 0
                if end_index > start_index:
                    bins.append(end_index- start_index)

        #logging.info("[Message] : bins :", bins)
        #logging.info("[Message] : mean(bins) :", np.mean(bins))
        moving_average_bin_w = int(np.median(bins)/2)
        print("[Message] :        bins :", bins)
        print("[Message] :   mean(bins) :", np.mean(bins))
        print("[Message] : median(bins) :", np.median(bins))

        projectArrary_average_1 = self.moving_average(projectArrary,moving_average_bin_w)
        #projectArrary_average_1 = np.concatenate((np.zeros(moving_average_bin_w/2),projectArrary_average_1))

        projectArrary_average_2 = self.moving_average(projectArrary_average_1, moving_average_bin_w)
        #projectArrary_average_2 = np.concatenate((np.zeros(moving_average_bin_w), projectArrary_average_2))

        #projectArrary_average = self.moving_medlian(projectArrary, 5)
        #self.array_show(projectArrary)
        #self.two_array_show_at_once(projectArrary, projectArrary_average_1)
        #self.two_array_show_at_once(projectArrary_average_1,projectArrary_average_2)

        #self.hist_show(projectArrary_average_1)
        #a = np.array((200,100,250,110,220))
        peaks,troughs = self.getPeaksAndTroughs(projectArrary_average_1,moving_average_bin_w/2)
        #peaks, troughs = self.getPeaksAndTroughs(a, 1)
        #self.array_extreme_point_show(a, 1)
        YProject_extreme_point_array = self.convertPeaksAndTroughsToArrary(projectArrary_average_1,peaks,projectArrary_threshold,troughs,projectArrary_threshold)

        #self.two_array_show_at_once(projectArrary_average_1,YProject_extreme_point_array)
        #print(YProject_extreme_point_array)
        #self.array_show(YProject_extreme_point_array)
        ##此处需要 对数组进行 波峰波谷的合并
        return self.peaksAndTroughsArraryPostProcess(YProject_extreme_point_array)


    def find_word_box(self,img_gray,img_colours):
        h, w = img_gray.shape
        y_project_array, y_project_img = self.YProject(img_gray)
        YProject_troughs_bins = self.YProjectPostProcess(y_project_array,0.3)
        #img_colours_draw = self.drawLineFromArraryToImg(img_colours,YProject_extreme_point_array)
        img_colours_draw = self.drawLineFromTroughsToImg(img_colours,YProject_troughs_bins)
        download_img(img_colours_draw, self.output_path, "05-img_gray_draw_line")

        #根据后处理出来的bins,在原图上一列一列的显示出来
        img_gray_h, img_gray_w = img_gray.shape


        index = 0
        for trough_index_start,trough_index_end in YProject_troughs_bins:
            if trough_index_start < 0 or trough_index_start > img_gray_w or  trough_index_end < 0 or trough_index_end > img_gray_w or trough_index_start >= trough_index_end:
                logging.warning(["[Waring] YProject_troughs_bin[",trough_index_start,",",trough_index_end,"] continue !!"])
                continue
            index=index+1

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
            download_img(img_gray_troughs_bin, self.output_path, "05-img_gray_YProject_troughs_bin_" + str(index))

    def hist_show(self,array):
        hist1, bins = np.histogram(array)  # hist1 每个灰度值的频数
        cdf = hist1.cumsum()  # 累加频数得累计直方图
        cdf_normalised = cdf * float(hist1.max() / cdf.max())  # 把累计直方图的比例化到近似直方图
        plt.plot(cdf_normalised, color='blue')
        plt.show()
    def array_show(self,array1):
        array1 = np.array(array1)
        max_val_array1 = np.max(array1)
        min_val_array1 = np.min(array1)
        array_len_array1 =  len(array1)
        #ax1= plt.subplot(2,1,1)
        array1_valid = array1[array1 > 0]
        array1_valid_sum = np.sum(array1_valid)
        array1_valid_max = np.max(array1_valid)
        array1_valid_median = np.median(array1_valid)
        array1_valid_mean = array1_valid_sum / len(array1_valid)

        array1_sum = np.sum(array1)
        array1_mean = array1_sum / len(array1)
        plt.plot(array1, label='array1', color='blue')
        plt.axhline(array1_valid_median,label='array1_valid_median',color='red')
        plt.axhline(array1_valid_mean,label='array1_valid_mean',color='green')
        plt.axhline(array1_mean, label='array1_mean', color='black')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.xlim(0, array_len_array1)
        plt.ylim(min_val_array1, max_val_array1)
        plt.xticks(np.arange(0, array_len_array1, (round(array_len_array1 / 100) * 10)))
        plt.yticks(np.arange(min_val_array1, max_val_array1, (round((max_val_array1-min_val_array1) / 100) * 10)))
        #plt.legend(loc=4)
        plt.title('array1')
        plt.show()
    def two_array_show_at_once(self,array1,array2):

        array1 = np.array(array1)
        max_val_array1   = max(np.max(array1),np.max(array2))
        array_len_array1 = max(len(array1),len(array2))

        plt.plot(array1, label='array1', color='blue')
        plt.plot(array2, label='array2', color='red')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.xlim(0, array_len_array1)
        plt.ylim(0, max_val_array1)
        plt.xticks(np.arange(0, array_len_array1, (round(array_len_array1 / 100) * 10)))
        plt.yticks(np.arange(0, max_val_array1, (round(max_val_array1 / 100) * 10)))
        #plt.legend(loc=4)
        plt.title('array')
        plt.show()
    def two_array_show(self,array1,array2):

        array1 = np.array(array1)
        max_val_array1 = np.max(array1)
        array_len_array1 =  len(array1)
        array1_valid = array1[array1 > 0]
        array1_valid_sum = np.sum(array1_valid)
        array1_valid_max = np.max(array1_valid)
        array1_valid_median = np.median(array1_valid)
        array1_valid_mean = array1_valid_sum / len(array1_valid)
        array1_sum = np.sum(array1)
        array1_mean = array1_sum / len(array1)

        ax1 = plt.subplot(2, 1, 1)
        plt.plot(array1, label='array1', color='blue')
        plt.axhline(array1_valid_median,label='array1_valid_median',color='red')
        plt.axhline(array1_valid_mean,label='array1_valid_mean',color='green')
        plt.axhline(array1_mean, label='array1_mean', color='black')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.xlim(0, array_len_array1)
        plt.ylim(0, max_val_array1)
        plt.xticks(np.arange(0, array_len_array1, (round(array_len_array1 / 100) * 10)))
        plt.yticks(np.arange(0, max_val_array1, (round(max_val_array1 / 100) * 10)))
        #plt.legend(loc=4)
        plt.title('array1')

        array2 = np.array(array2)
        max_val_array2 = np.max(array2)
        array_len_array2 = len(array2)
        array2_valid = array2[array2 > 0]
        array2_valid_sum = np.sum(array2_valid)
        array2_valid_max = np.max(array2_valid)
        array2_valid_median = np.median(array2_valid)
        array2_valid_mean = array2_valid_sum / len(array2_valid)
        array2_sum = np.sum(array2)
        array2_mean = array1_sum / len(array2)

        ax1 = plt.subplot(2, 1, 2)
        plt.plot(array2, label='array2', color='blue')
        plt.axhline(array2_valid_median, label='array1_valid_median', color='red')
        plt.axhline(array2_valid_mean, label='array1_valid_mean', color='green')
        plt.axhline(array2_mean, label='array1_mean', color='black')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.xlim(0, array_len_array2)
        plt.ylim(0, max_val_array2)
        plt.xticks(np.arange(0, array_len_array1, (round(array_len_array2 / 100) * 10)))
        plt.yticks(np.arange(0, max_val_array1, (round(max_val_array2 / 100) * 10)))
        # plt.legend(loc=4)
        plt.title('array2')
        plt.show()
    def array_extreme_point_show(self,array1,distance_th):
        array1 = np.array(array1)
        max_val_array1 = np.max(array1)
        array_len_array1 =  len(array1)
        #ax1= plt.subplot(2,1,1)
        array1_valid = array1[array1 > 0]
        array1_valid_sum = np.sum(array1_valid)
        array1_valid_max = np.max(array1_valid)
        array1_valid_median = np.median(array1_valid)
        array1_valid_mean = array1_valid_sum / len(array1_valid)

        array1_sum = np.sum(array1)
        array1_mean = array1_sum / len(array1)
        plt.plot(array1, label='array1', color='blue')
        plt.axhline(array1_valid_median,label='array1_valid_median',color='red')
        plt.axhline(array1_valid_mean,label='array1_valid_mean',color='green')
        plt.axhline(array1_mean, label='array1_mean', color='black')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.xlim(0, array_len_array1)
        plt.ylim(0, max_val_array1)
        #plt.xticks(np.arange(0, array_len_array1, (round(array_len_array1 / 100) * 10)))
        #plt.yticks(np.arange(0, max_val_array1, (round(max_val_array1 / 100) * 10)))
        #plt.legend(loc=4)
        plt.title('array1')
        peaks,troughs = self.getPeaksAndTroughs(array1, distance_th)  # distance表极大值点的距离至少大于等于10个水平单位
        #peaks = peaks.astype(np.uint32)
        #troughs = troughs.astype(np.uint32)
        print('the number of peaks is ' + str(len(peaks)))
        for x, y in peaks:
            plt.text(x, y, y, fontsize=10, verticalalignment="bottom", horizontalalignment="center")
        for x, y in troughs:
            plt.text(x, y, y, fontsize=10, verticalalignment="top", horizontalalignment="center")
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
        logging.info("##############################################")
        crop_out_path = output_path + "/" + img_path
        crop_out_name = os.path.splitext(crop_out_path)[0]

        worker = CROP_WORD(crop_out_name)

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

