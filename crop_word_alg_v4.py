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
    is_auto_y_project_threshold = True
    y_project_threshold_scale = 0.5
    y_projection_array_th_img = []

    y_bins = []
    y_bin_width_threshold = 20
    y_bins_dis_threshold = 5
    y_bins_expan_width = 5




    x_bins = []
    x_project_threshold = 3
    x_bin_width_threshold = 10
    x_bins_dis_threshold = 3
    x_bins_expan_width =   5

    bins = []

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

    def setYProjectThreshold(self,is_auto_y_project_threshold,threshold):
        self.is_auto_y_project_threshold = is_auto_y_project_threshold
        if is_auto_y_project_threshold == False :
            self.y_project_threshold = threshold
            logging.info("[Waring] change y_project_threshold  = %s", self.y_project_threshold)


    def getYProjectThresholdScale(self):
        return self.y_project_threshold_scale

    def setYProjectThresholdScale(self, scale):
        self.y_project_threshold_scale = scale

    def getYBinsExpanWidth(self):
        return self.y_bins_expan_width

    def setYBinsExpanWidth(self, y_bins_expan_width):
        self.y_bins_expan_width = y_bins_expan_width

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

    def cleanState(self):
        self.output_middle_path = "./output_middle"

        self.is_handwriting_black = False
        self.is_auto_threshold = True

        self.img_gray_threshold = 0

        self.y_projection_img = []
        self.y_project_threshold = 0
        self.is_auto_y_project_threshold = True
        self.y_project_threshold_scale = 0.5
        self.y_projection_array_th_img = []

        self.y_bins = []
        self.y_bin_width_threshold = 20
        self.y_bins_dis_threshold = 5
        self.y_bins_expan_width = 5


    #????????????
    def moving_average(self,x, w):
        tmp = np.convolve(x, np.ones(w), 'valid') / w
        return np.concatenate((np.zeros(math.ceil(w / 2)), tmp))

    def moving_medlian(self,x, w):
        return scipy.signal.medfilt(x, w)

    def arrayThreshold(self,array,th):
        tmp = array.copy()
        tmp[tmp < (th)] = 0
        return tmp
    def clearMiddleDirs(self):
        if True == os.path.exists(self.output_middle_path):
            remove_dirs(self.output_middle_path)


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

        # 00.??????????????????
        input = cv2.GaussianBlur(input, (5, 5), 0)
        download_img(input, self.output_middle_path, "01-GaussianBlur")

        img_gray = input
        if input.ndim == 3:
            img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        download_img(img_gray, self.output_middle_path, "02-gray")
        height, width = img_gray.shape  # ??????????????????

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
        # ??????????????????
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

    # ???????????????
    def YProject(self,binary):
        y_projection_array = np.sum(binary, axis=0) /255
        y_projection_array = self.moving_average(y_projection_array,5)
        return  np.array(y_projection_array)

    def converYProjectToImg(self,y_projection_array):
        y_projection_array_max = np.max(y_projection_array).astype(int)

        #?????????,?????????????????????0
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

    def cleanBinsFlagFalse(self,bins):
        result = []
        for bin in bins:
            bin_start, bin_end, flag = bin
            if flag < 0:
                continue
            result.append(bin)
        return result

    def binsAve(self, bins):
        sum = 0
        index = 0
        for bin_start, bin_end,flag in bins:
            if flag < 0 :
                continue
            sum = sum + (bin_end - bin_start)
            index = index + 1
        if sum == 0:
            return 0
        else:
            return sum / index

    def binsPostProcess(self,project_array,bins,bin_width_threshold,bins_dis_threshold):
        project_array_width, = project_array.shape
        #?????????????????????????????????
        for i in range(len(bins)):
            bin_start, bin_end,flag = bins[i]
            if flag < 0:
                continue

            if i != 0 :  #?????????????????????

                bin_befre_start, bin_befre_end,flag = bins[i - 1]

                # ???????????????????????????,??????????????????,??????????????????.??????????????????,????????????.???????????????
                if bin_start - bin_befre_end > bins_dis_threshold :
                    continue
                bins[i - 1] = (0,0,-1)
                bins[i] = (bin_befre_start, bin_end,1)

        bins = self.cleanBinsFlagFalse(bins)


        #????????????????????????
        for i in range(len(bins)):
            bin_start, bin_end,flag = bins[i]
            if flag < 0 :
                continue
            if (bin_end - bin_start) <= bin_width_threshold:
                befre_i_index = -1
                for befre_i in range(i - 1 ,-1,-1):
                    befre_i_start, befre_i_end, befre_i_flag = bins[befre_i]
                    if befre_i_flag == 1:
                        befre_i_index = befre_i
                        break

                if befre_i_index == -1 and i != len(bins) -1:  #???????????????,????????????
                    bin_next_start, bin_next_end,flag = bins[i+1]
                    bins[i+1] = (bin_start,bin_next_end,1)
                if befre_i_index != -1 and i == len(bins) - 1:  # ????????????,???????????????
                    bin_befre_start, bin_befre_en,flag = bins[befre_i_index]
                    bins[befre_i_index] = (bin_befre_start,bin_end,1)
                if befre_i_index != -1 and i != len(bins) - 1:  # ???????????????

                    #????????????????????????????????????????????????,?????????????????????????????????????????????????????????,????????????
                    project_array_bin_crop = project_array[bin_start:bin_end]
                    project_array_bin_crop_max = np.max(project_array_bin_crop)
                    project_array_bin_crop_max_indexs = np.where(project_array_bin_crop_max)
                    bin_meddle = int(bin_start + np.mean(project_array_bin_crop_max_indexs))

                    project_array_bin_crop_meddle_left = project_array[bin_start:bin_meddle]
                    project_array_bin_crop_meddle_right = project_array[bin_meddle:bin_end]
                    project_array_bin_crop_meddle_left_sum = np.sum(project_array_bin_crop_meddle_left)
                    project_array_bin_crop_meddle_right_sum = np.sum(project_array_bin_crop_meddle_right)

                    bin_befre_start, bin_befre_en, flag = bins[befre_i_index]
                    bin_next_start, bin_next_end, flag = bins[i + 1]
                    dis_befre = bin_meddle - bin_befre_end
                    dis_next = bin_next_start - bin_meddle
                    if project_array_bin_crop_meddle_left_sum > project_array_bin_crop_meddle_right_sum :
                        bins[befre_i_index] = (bin_befre_start, bin_end, 1)
                    elif project_array_bin_crop_meddle_left_sum <project_array_bin_crop_meddle_right_sum :
                        bins[i + 1] = (bin_start, bin_next_end,1)
                    else:
                        if dis_befre < dis_next :
                            bins[befre_i_index] = (bin_befre_start, bin_end, 1)
                        elif dis_befre > dis_next :
                            bins[i + 1] = (bin_start, bin_next_end, 1)
                        else:
                            bins[i + 1] = (bin_start, bin_next_end, 1)

                bins[i] = (0,0,-1)

        bins = self.cleanBinsFlagFalse(bins)

        bins_ave = self.binsAve(bins)



        #???bins??????
        for i in range(len(bins) -1):
            bin_start, bin_end,flag = bins[i]
            if flag < 0 :
                continue
            if i !=  len(bins):
                bin_next_start, bin_next_end,bin_next_flag = bins[i+1]
                #if bin_next_start - bin_end > bins_ave:
                #    bins[i] = (bin_start, bin_end,2)
                #    continue

                #project_array_bin_crop = project_array[bin_end:bin_next_start]
                #project_array_bin_crop_min = np.min(project_array_bin_crop)
                #project_array_bin_crop_min_indexs = np.where(project_array_bin_crop_min)
                #through_meddle = int(bin_end + np.mean(project_array_bin_crop_max_indexs))

                #???????????????,??????????????????????????????,????????????????????????????????????.????????????????????????
                through_meddle = int((bin_end + bin_next_start ) /2)

                bins[i] = (bin_start, through_meddle,flag)
                bins[i+1] = (through_meddle, bin_next_end,bin_next_flag)


        bins = self.cleanBinsFlagFalse(bins)

        # ??????????????????????????????bin?????????,??????????????????????????????????????????????????????
        if len(bins)  == 1:
            bin_start, bin_end,flag = bins[0]
            bins[0] = (max(0,bin_start - self.y_bins_expan_width),bin_end,flag)
        elif len(bins)  >=2 :
            bin_start, bin_end,flag = bins[0]
            bins[0] = (max(0,bin_start - self.y_bins_expan_width),bin_end,flag)
            bin_start_2, bin_end_2,flag_2 = bins[-1]
            bins[-1] = (bin_start_2,min(bin_end_2 + self.y_bins_expan_width,int(project_array_width)-1),flag)

        return bins

    def drawLineFromYProjectBinsToImg(self, img_colours, bins):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_tmp = img_colours.copy()
        index = 0
        for trough_index_start,trough_index_end,flag in bins:
            if flag < 0 :
                continue
            cv2.line(img_tmp, (trough_index_start, 0), (trough_index_start, img_gray_h),   (0,255,0),2)
            cv2.line(img_tmp, (trough_index_end, 0), (trough_index_end, img_gray_h),       (0,0,255),2)
            cv2.line(img_tmp, (trough_index_start, int(img_gray_h/4)), (trough_index_end, int(img_gray_h/4)), (0, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX  # ????????????
            #img_tmp = cv2.putText(img_tmp, str(index), (trough_index_start, int(img_gray_h/4)), font, 10, (0, 0, 0), 1)
            index=index+1
        return img_tmp

    def drawLineFromYProjectThresholdToImg(self, img_colours, threshold):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_tmp = img_colours.copy()
        threshold = int(min(threshold,img_gray_h))
        cv2.line(img_tmp, (0, img_gray_h - threshold), (img_gray_w, img_gray_h - threshold), (0, 0, 255), 3)
        return img_tmp


    # ??????????????????
    def XProject(self,binary):
        x_projection_array = np.sum(binary, axis=1) /255
        x_projection_array = self.moving_average(x_projection_array,5)
        return x_projection_array

    def cropYProjectBinsToImg(self,img_color,img_gray,bins):
        img_gray_h, img_gray_w = img_gray.shape
        img_tmp = img_gray.copy()
        index = 0
        bins_and_img = []
        for bin_start, bin_end,flag in bins:
            if flag < 0 :
                continue
            img_gray_crop = img_gray[:, bin_start:bin_end]
            img_color_crop = img_color[:, bin_start:bin_end]
            bins_and_img.append((bin_start,bin_end,flag,img_gray_crop,img_color_crop))
            download_img(img_color_crop, self.output_middle_path, "10-img_y_project_crop_" + str(index))
            index = index + 1
        return bins_and_img
    def XProjectPostProcessToBins(self,):

        return
    def converXProjectToImg(self,x_projection_array):
        x_projection_array_max = np.max(x_projection_array).astype(int)

        #?????????,?????????????????????0
        x_projection_array_max = max(20,x_projection_array_max)
        x_projection_img = np.zeros((len(x_projection_array),int(x_projection_array_max), 3), dtype=np.uint8)
        x_projection_img[:] = 255
        for i in range(len(x_projection_array)):
            if x_projection_array[i] != 0:
                cv2.line(x_projection_img, (int(x_projection_array_max - x_projection_array[i]),i ),
                         (int(x_projection_array_max),i), (255, 0, 0))
        return x_projection_img

    def drawLineFromXProjectBinsToImg(self, img_colours, bins):
        img_gray_h, img_gray_w,img_gray_c = img_colours.shape
        img_tmp = img_colours.copy()

        for trough_index_start,trough_index_end,flag in bins:
            if flag < 0 :
                continue
            cv2.line(img_tmp, (0,trough_index_start), (img_gray_w,trough_index_start),   (0,255,0),2)
            cv2.line(img_tmp, (0,trough_index_end), (img_gray_w,trough_index_end),       (0,0,255),2)
            cv2.line(img_tmp, (int(img_gray_w/4),trough_index_start), (int(img_gray_w/4),trough_index_end), (0, 0, 0), 2)

        return img_tmp

    def porcessYProjectBinsAndImg(self,bins_and_img):

        bins = []
        index = 0
        for y_bin_start, y_bin_end, flag, y_project_img_gray_crop,y_project_img_color_crop in bins_and_img:
            x_projection_array = self.XProject(y_project_img_gray_crop)
            x_projection_array_img = self.converXProjectToImg(x_projection_array)
            download_img(x_projection_array_img, self.output_middle_path, "10-img_y_project_crop_" + str(index) + "_x_projection_array_img")

            x_projection_array_th = self.arrayThreshold(x_projection_array,self.x_project_threshold)
            x_projection_array_th_img = self.converXProjectToImg(x_projection_array_th)
            download_img(x_projection_array_th_img, self.output_middle_path,
                         "10-img_y_project_crop_" + str(index) + "_x_projection_array_th_img")

            # ?????????????????????bins
            x_bins = self.YProjectPostProcessToBins(x_projection_array_th)
            x_bins = self.binsPostProcess(x_projection_array_th, x_bins, self.x_bin_width_threshold, self.x_bins_dis_threshold)
            x_bins_img = self.drawLineFromXProjectBinsToImg(y_project_img_color_crop,x_bins)
            download_img(x_bins_img, self.output_middle_path,"10-img_y_project_crop_" + str(index) + "_x_bins_img")
            for x_bin in x_bins:
                x_bin_start, x_bin_end, flag  =  x_bin
                bins.append((y_bin_start,y_bin_end,x_bin_start,x_bin_end))
            #?????????
            index = index + 1
        return bins

    def converToBinsFromSignPeakArray(self,array):
        peaks_satrt = np.where(array == -1)
        peaks_end   = np.where(array == -10)
        peaks_w = peaks_end - peaks_satrt
        peaks_w[peaks_w <= 0] = 0
        return np.average(peaks_w)

    def auto_work(self,input):
        #?????????
        self.img_gray_pre_processed = self.pre_process(input,self.is_auto_threshold,self.is_handwriting_black)

        #????????????
        self.y_projection_array =  self.YProject(self.img_gray_pre_processed)

        #???????????????img
        self.y_projection_img = self.converYProjectToImg(self.y_projection_array)
        download_img(self.y_projection_img, self.output_middle_path, "06-y_projection_img")

        #????????????????????????????????????,?????????????????????
        if self.is_auto_y_project_threshold == True:
            self.y_project_threshold = self.computeYProjectThreshold(self.y_projection_array)

        #????????????????????????????????????????????????
        self.y_projection_array_th = self.arrayThreshold(self.y_projection_array,self.y_project_threshold * self.y_project_threshold_scale)

        #???????????????????????????????????????img
        self.y_projection_array_th_img = self.converYProjectToImg(self.y_projection_array_th)
        download_img(self.y_projection_array_th_img, self.output_middle_path, "07-y_projection_array_th_img")

        #?????????????????????bins
        self.y_bins = self.YProjectPostProcessToBins(self.y_projection_array_th)

        #??????bins???????????????????????????????????????
        #??????????????????

        #????????????bins?????????,??????ok
        self.y_bins = self.binsPostProcess(self.y_projection_array_th,self.y_bins,self.y_bin_width_threshold,self.y_bins_dis_threshold)

        #??????????????????bins???????????????
        self.y_projection_img = self.drawLineFromYProjectBinsToImg(self.y_projection_img,self.y_bins)
        self.y_projection_img = self.drawLineFromYProjectThresholdToImg(self.y_projection_img,self.y_project_threshold)
        self.y_projection_array_th_img = self.drawLineFromYProjectBinsToImg(self.y_projection_array_th_img, self.y_bins)



        #????????????????????????????????????
        self.y_project_bins_and_img = self.cropYProjectBinsToImg(input,self.img_gray_pre_processed,self.y_bins)
        self.porcessYProjectBinsAndImg(self.y_project_bins_and_img )





        return


