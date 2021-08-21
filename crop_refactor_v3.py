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


    def __init__(self):
        super().__init__()
        self.iconPath = "./resource/icon.jpg"
        desktop = QApplication.desktop()

        self.window_w = desktop.width()
        self.window_h = desktop.height()
        self.desktop_scale = 0.9
        self.img_windows_scale = 0.7
        self.initUI()
    def initUI(self):
        self.setWindowTitle('汉字书法切割工具')
        self.setWindowIcon(QIcon(self.iconPath))
        self.setFixedSize(self.window_w * self.desktop_scale, self.window_h * self.desktop_scale)

        #按钮
        QToolTip.setFont(QFont('SansSerif', 12))
        self.boutton1 = QPushButton('1.导入单张图片', self)
        self.boutton1.setToolTip('选择需要导入的图片的路径')
        self.boutton1.resize(self.boutton1.sizeHint())
        self.boutton1.move(20, 30)
        self.boutton1.clicked.connect(self.exportOneImgPathFile)

        self.boutton2 = QPushButton('2.自动分割', self)
        self.boutton2.setToolTip('全自动分割书法字帖的字.\t只适用于背景单一场景简单的字帖')
        self.boutton2.resize(self.boutton2.sizeHint())
        self.boutton2.move(20, 60)
        self.boutton2.clicked.connect(self.autoCropWord)



        #显示图片的区域
        self.label1 = ShowImgLabel(self)
        self.label1.setText("显示图片")
        self.label1.setFixedSize(self.window_w * self.img_windows_scale, self.window_h * self.img_windows_scale)
        self.label1.move(120, 75)

        self.label1.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:50px;font-weight:bold;font-family:宋体;}"
                              )

        #显示垂直投影区域
        self.label2 = QLabel(self)
        self.label2.setText("显示垂直投影")
        self.label2.setFixedSize(self.window_w * self.img_windows_scale, 100)
        self.label2.move(120, 850)
        self.label2.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:50px;font-weight:bold;font-family:宋体;}"
                              )
        #显示垂直投影区域
        self.label3 = QLabel(self)
        self.label3.setText("水平投影")
        self.label3.setFixedSize(150, self.window_h * self.img_windows_scale)
        self.label3.move(1500, 75)
        self.label3.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}"
                              )

        #显示图片路径
        self.label4 = QLabel(self)
        self.label4.setText("图片路径")
        self.label4.setFixedSize(self.window_w * self.img_windows_scale, 45)
        self.label4.move(120, 20)
        self.label4.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                              )




        self.center()
        self.show()

    def exportOneImgPathFile(self):
        root = tk.Tk()
        root.withdraw()  # 隐藏Tk窗口
        imgPath = filedialog.askopenfilename()
        self.label4.setText(imgPath)
        self.show_img(imgPath)
        #print(imgPath)
    def autoCropWord(self):
        if self.isLoadImg == False :
            reply = QMessageBox.question(self, '错误!',
                                         "没有导入图片!", QMessageBox.Yes , QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return
        logging.info("[Message] doing autoCropWord")



    def closeEvent(self, event):

        reply = QMessageBox.question(self, '消息框标题', '你确定要退出吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def show_img1(self, image_path):
        self.QtImg = QtGui.QPixmap(image_path)
        self.label1.setPixmap(self.QtImg)
    def show_img(self,image_path):
        self.QtImg = QtGui.QPixmap(image_path)

        self.QtImgWidthScale  =  self.label1.width() /self.QtImg.width()
        self.QtImgHeightScale =  self.label1.height() /self.QtImg.height()

        #if self.QtImgWidthScale  < 1 or self.QtImgHeightScale  < 1:
        self.QtImgScale  = min(self.QtImgWidthScale,self.QtImgHeightScale)
        print("QtImgScale = ",self.QtImgScale)
        self.label1.setScaledContents(False)
        pix = self.QtImg.scaled(int(self.QtImg.width()*self.QtImgScale),int(self.QtImg.height()*self.QtImgScale), aspectRatioMode=Qt.KeepAspectRatio,transformMode=Qt.SmoothTransformation)
        self.label1.setPixmap(pix)
        self.isLoadImg = True
        return
        ##后边没有用
        if self.QtImgScale != self.QtImgWidthScale:
        # 根据label宽度等比例缩放图片
        #ui->imgLable->setPixmap(pix.scaledToWidth(ui->imgLable->width()));

            dst = self.QtImg.scaledToWidth(self.label1.width(), aspectRatioMode=Qt.KeepAspectRatio)
            print("pix w = "    , dst.width())
            print("pix h = "    , dst.height())
            self.label1.setPixmap(dst)
        #// 根据label高度等比例缩放图片
        #ui->imgLable->setPixmap(pix.scaledToHeight(ui->imgLable->height()));
        else:
            self.label1.setPixmap(self.QtImg.scaledToHeight(self.label1.height(),aspectRatioMode=Qt.KeepAspectRatio))
        #self.label1.setPixmap(self.QtImg.scaled(int(self.QtImg.width() *self.QtImgWidthScale),int(self.QtImg.height()*self.QtImgHeightScale),Qt.aspectRatioMode,Qt.SmoothTransformation))
        #self.label1.setPixmap(pix)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == '__main__':
    #filedialog.askopenfilename()
    app = QApplication(sys.argv)
    ex = CROP_WORD_APP()
    sys.exit(app.exec_())