import logging

import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import tkinter as tk
from tkinter import filedialog
from util import *
from crop_word_alg_v4 import *

class ShowImgLabel(QLabel):
    rects = []

    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0

    min_x = 0
    min_y = 0

    max_x = 0
    max_y = 0

    flag = False
    # 鼠标点击事件
    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.flag = False
        self.rects.append(self.getRecct())

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.min_x = min(self.x0, self.x1)
            self.max_x = max(self.x0, self.x1)
            self.min_y = min(self.y0, self.y1)
            self.max_y = max(self.y0, self.y1)
            self.update()

    # 绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)

        rect = QRect(self.min_x, self.min_y, abs(self.max_x - self.min_x), abs(self.max_y- self.min_y))
        painter = QPainter(self)
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        painter.drawRect(rect)

        for rect in self.rects:
            painter.drawRect(rect)

    def getRecct(self):
        return QRect(self.min_x, self.min_y, abs(self.max_x - self.min_x), abs(self.max_y- self.min_y))

    def getRects(self):
        return self.rects

    def clearRects(self):
        self.rects = []
        self.flag = False
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0



class CROP_WORD_APP(QWidget):
    isLoadImg = False


    def __init__(self,CROP_WORD_ALG):
        super().__init__()

        desktop = QApplication.desktop()
        self.window_w = desktop.width()
        self.window_h = desktop.height()
        self.desktop_scale = 0.93
        self.img_windows_scale = 0.7
        self.iconPath = "./resource/icon.jpg"
        self.cropWordAlg = CROP_WORD_ALG
        self.initUI()
    def initUI(self):
        self.setWindowTitle('汉字书法切割工具')
        self.setWindowIcon(QIcon(self.iconPath))
        self.setFixedSize(self.window_w * self.desktop_scale, self.window_h * self.desktop_scale)

        #####################################################################
        #按钮:导入单张图片
        QToolTip.setFont(QFont('SansSerif', 12))
        self.boutton1 = QPushButton('1.导入单张图片', self)
        self.boutton1.setToolTip('选择需要导入的图片的路径')
        self.boutton1.resize(self.boutton1.sizeHint())
        self.boutton1.move(20, 45)
        self.boutton1.clicked.connect(self.exportOneImgPathFile)

        #####################################################################
        # 按钮:设置手写汉字是黑色还是白色
        self.boutton2_1 = QRadioButton(self)
        self.boutton2_2 = QRadioButton(self)
        self.boutton2Group = QButtonGroup(self)
        self.boutton2Group.addButton(self.boutton2_1)
        self.boutton2Group.addButton(self.boutton2_2)
        self.boutton2_1.setText("黑色汉字")
        self.boutton2_1.move(20, 90)
        self.boutton2_2.setText("白色汉字")
        self.boutton2_2.move(20, 110)
        self.boutton2_1.setChecked(True)
        self.boutton2_1.setToolTip('选择汉字是黑色还是白色.需要此设置将图像阈值化为黑色为背景,汉字为前景的二值图像')
        self.boutton2_2.setToolTip('选择汉字是黑色还是白色.需要此设置将图像阈值化为黑色为背景,汉字为前景的二值图像')
        #####################################################################
        # 按钮:自动分割
        self.boutton3 = QPushButton('2.自动分割', self)
        self.boutton3.setToolTip('全自动分割书法字帖的字.\n只适用于背景单一场景简单的字帖')
        self.boutton3.resize(self.boutton3.sizeHint())
        self.boutton3.move(20, 300)
        self.boutton3.clicked.connect(self.autoCropWord)

        #####################################################################
        # 按钮:固定阈值还是自动阈值
        self.boutton4_1 = QRadioButton(self)
        self.boutton4_2 = QRadioButton(self)
        self.boutton4Group = QButtonGroup(self)
        self.boutton4Group.addButton(self.boutton4_1)
        self.boutton4Group.addButton(self.boutton4_2)
        self.boutton4_1.setText("自动阈值")
        self.boutton4_1.move(20, 150)
        self.boutton4_2.setText("固定阈值")
        self.boutton4_2.move(20, 190)
        #self.boutton4_1.setChecked(True)
        self.boutton4_2.setChecked(True)
        self.boutton4_1.setToolTip('选择是自动计算阈值还是固定阈值')
        self.boutton4_2.setToolTip('选择是自动计算阈值还是固定阈值')

        #####################################################################
        #标签:显示自动阈值
        self.lineEdit0 = QLineEdit(self)
        #self.label0.set
        self.lineEdit0.setFixedSize(80,20)
        self.lineEdit0.move(20, 170)
        self.lineEdit0.setToolTip('显示自动阈值')
        self.lineEdit0.setAlignment(Qt.AlignLeft)
        self.lineEdit0.setReadOnly(True)

        #####################################################################
        #标签:显示固定阈值
        self.lineEdit1 = QLineEdit(self)
        self.lineEdit1.setText("127")
        self.lineEdit1.setFixedSize(80,20)
        self.lineEdit1.move(20, 210)
        self.lineEdit1.setToolTip('显示or设置固定阈值.只能是整数!')
        self.lineEdit1.setValidator(QIntValidator())
        self.lineEdit1.setAlignment(Qt.AlignLeft)



        #####################################################################
        #标签:显示图片路径
        self.label1 = QLabel(self)
        self.label1.setText("图片路径区域")
        self.label1.setFixedSize(self.window_w * self.img_windows_scale, 35)
        self.label1.move(120, 5)
        self.label1.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                  )

        #####################################################################
        #标签:显示图片的区域
        self.label2 = ShowImgLabel(self)
        self.label2.setText("显示图片区域")
        self.label2.setFixedSize(self.window_w * self.img_windows_scale, self.window_h * self.img_windows_scale)
        self.label2.move(120, 45)

        self.label2.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:50px;font-weight:bold;font-family:宋体;}"
                                  )

        #####################################################################
        #标签:显示垂直投影区域
        self.label3 = QLabel(self)
        self.label3.setText("显示垂直投影区域")
        self.label3.setFixedSize(self.window_w * self.img_windows_scale, 95)
        self.label3.move(120, 810)
        self.label3.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:50px;font-weight:bold;font-family:宋体;}"
                                  )

        #####################################################################
        #标签:显示垂直投影阈值化后区域
        self.label4 = QLabel(self)
        self.label4.setText("显示垂直投影阈值化后区域")
        self.label4.setFixedSize(self.window_w * self.img_windows_scale, 95)
        self.label4.move(120, 910)
        self.label4.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:50px;font-weight:bold;font-family:宋体;}"
                                  )





        #####################################################################
        #标签:显示水平投影区域
        self.label10 = QLabel(self)
        self.label10.setText("水平投影")
        self.label10.setFixedSize(150, self.window_h * self.img_windows_scale)
        self.label10.move(1500, 45)
        self.label10.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}"
                                   )





        self.center()
        self.show()


    def show_img_opencv(self,img_opencv):

        #这里的格式需要注意 :COLOR_BGR2BGRA
        self.QtImg = QImage(cv2.cvtColor(img_opencv,cv2.COLOR_BGR2BGRA), img_opencv.shape[1], img_opencv.shape[0], QImage.Format_RGB32)
        #self.QtImg.save("./222.jpg","JPG",-1)
        self.QtImgWidthScale  = self.label2.width() / self.QtImg.width()
        self.QtImgHeightScale = self.label2.height() / self.QtImg.height()

        #if self.QtImgWidthScale  < 1 or self.QtImgHeightScale  < 1:
        self.QtImgScale  = min(self.QtImgWidthScale,self.QtImgHeightScale)
        self.label2.setScaledContents(False)
        pix = self.QtImg.scaled(int(self.QtImg.width()*self.QtImgScale),int(self.QtImg.height()*self.QtImgScale), aspectRatioMode=Qt.KeepAspectRatio,transformMode=Qt.SmoothTransformation)
        self.label2.setPixmap(QPixmap.fromImage(pix))
        self.isLoadImg = True

    def exportOneImgPathFile(self):
        root = tk.Tk()
        root.withdraw()  # 隐藏Tk窗口
        imgPath = filedialog.askopenfilename()
        self.label1.setText(imgPath)
        self.img_opencv = cv_imread(imgPath)
        if self.img_opencv is None:
            reply = QMessageBox.question(self, '错误!',
                                         "读取图片失败!", QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return

        self.show_img_opencv(self.img_opencv)

    def show_img_opencv_YProject(self, img_opencv):

        # 这里的格式需要注意 :COLOR_BGR2BGRA
        self.QtImgYProject = QImage(cv2.cvtColor(img_opencv, cv2.COLOR_BGR2BGRA), img_opencv.shape[1], img_opencv.shape[0],
                            QImage.Format_RGB32)
        #self.label2.setScaledContents(True)

        self.QtImgYProjectHeightScale = self.label3.height() / self.QtImgYProject.height()
        pix = self.QtImgYProject.scaled(int(self.QtImgYProject.width() * self.QtImgScale), int(self.QtImgYProject.height() * self.QtImgYProjectHeightScale),transformMode=Qt.SmoothTransformation)
        #pix = pix.scaledToWidth(int(self.QtImgYProject.width() * self.QtImgScale))
        #pix.save("./222.jpg","JPG",-1)
        self.label3.setPixmap(QPixmap.fromImage(pix))

    def show_img_opencv_YProjectTh(self, img_opencv):

        # 这里的格式需要注意 :COLOR_BGR2BGRA
        self.QtImgYProject = QImage(cv2.cvtColor(img_opencv, cv2.COLOR_BGR2BGRA), img_opencv.shape[1], img_opencv.shape[0],
                            QImage.Format_RGB32)
        #self.label2.setScaledContents(True)

        self.QtImgYProjectHeightScale = self.label3.height() / self.QtImgYProject.height()
        pix = self.QtImgYProject.scaled(int(self.QtImgYProject.width() * self.QtImgScale), int(self.QtImgYProject.height() * self.QtImgYProjectHeightScale),transformMode=Qt.SmoothTransformation)
        #pix = pix.scaledToWidth(int(self.QtImgYProject.width() * self.QtImgScale))
        #pix.save("./222.jpg","JPG",-1)
        self.label4.setPixmap(QPixmap.fromImage(pix))

    def autoCropWord(self):

        #此处需要将各个lable清除

        if self.isLoadImg == False :
            reply = QMessageBox.question(self, '错误!',
                                         "没有导入图片!", QMessageBox.Yes , QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return
        logging.info("[Message] doing autoCropWord")
        self.cropWordAlg.setHandWritingIsBlack(self.boutton2_1.isChecked())
        isAutoThreshold = self.boutton4_1.isChecked()
        if isAutoThreshold ==True:
            self.cropWordAlg.setIsAutoThreshold(True)
        else:
            self.cropWordAlg.setIsAutoThreshold(False,int(self.lineEdit1.text()))
        self.img_opencv_y_projection,self.img_opencv_y_projection_th = self.cropWordAlg.auto_work(self.img_opencv)
        self.lineEdit0.setText(str(self.cropWordAlg.getImgGrayThreshold()))
        self.show_img_opencv_YProject(self.img_opencv_y_projection)
        self.show_img_opencv_YProjectTh(self.img_opencv_y_projection_th)




    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):

        reply = QMessageBox.question(self, '消息框标题', '你确定要退出吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    #filedialog.askopenfilename()
    output_middle_path = "./output_middle"
    output_path = "./output"

    app = QApplication(sys.argv)
    crop_word_alg_v4 = CROP_WORD_ALG()
    crop_word_alg_v4.setOutputMiddlePath(output_middle_path)
    ex = CROP_WORD_APP(crop_word_alg_v4)

    sys.exit(app.exec_())