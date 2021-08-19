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



class CROP_WORD(QWidget):
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
        self.boutton1 = QPushButton('导入单张图片', self)
        self.boutton1.setToolTip('选择需要导入的图片的路径')
        self.boutton1.resize(self.boutton1.sizeHint())
        self.boutton1.move(20, 30)
        self.boutton1.clicked.connect(self.exportImgPathFile)

        #显示垂直投影区域
        self.label4 = QLabel(self)
        self.label4.setText("图片路径")
        self.label4.setFixedSize(self.window_w * self.img_windows_scale, 30)
        self.label4.move(120, 30)
        self.label4.setStyleSheet("QLabel{background:white;}"
                              "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}"
                              )

        #显示图片的区域
        self.label1 = QLabel(self)
        self.label1.setText("显示图片")
        self.label1.setFixedSize(self.window_w * self.img_windows_scale, self.window_h * self.img_windows_scale)
        self.label1.move(120, 75)
        self.label1.setScaledContents(True)
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



        self.center()
        self.show()

    def exportImgPathFile(self):
        root = tk.Tk()
        root.withdraw()  # 隐藏Tk窗口
        imgPath = filedialog.askopenfilename()
        self.show_img(imgPath)
        #print(imgPath)

    def closeEvent(self, event):

        reply = QMessageBox.question(self, '消息框标题', '你确定要退出吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def show_img(self,image_path):


        jpg = QtGui.QPixmap(image_path).scaledToWidth(self.label1.width())
        self.label1.setPixmap(jpg)


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == '__main__':
    #filedialog.askopenfilename()
    app = QApplication(sys.argv)
    ex = CROP_WORD()
    sys.exit(app.exec_())