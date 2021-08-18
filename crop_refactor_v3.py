import logging

import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
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
        self.window_scale = 0.9
        self.initUI()
    def initUI(self):
        self.setWindowTitle('汉字书法切割工具')
        self.setWindowIcon(QIcon(self.iconPath))
        self.setFixedSize(self.window_w * self.window_scale, self.window_h * self.window_scale)

        #按钮
        QToolTip.setFont(QFont('SansSerif', 12))
        self.boutton1 = QPushButton('导入单张图片', self)
        self.boutton1.setToolTip('选择需要导入的图片的路径')
        self.boutton1.resize(self.boutton1.sizeHint())
        self.boutton1.move(20, 40)
        self.boutton1.clicked.connect(self.exportImgPathFile)

        self.center()

        self.show()
    def exportImgPathFile(self):
        root = tk.Tk()
        root.withdraw()  # 隐藏Tk窗口
        imgPath = filedialog.askopenfilename()
        #print(imgPath)

    def closeEvent(self, event):

        reply = QMessageBox.question(self, '消息框标题', '你确定要退出吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

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