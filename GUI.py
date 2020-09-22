# -*- coding: utf-8 -*-
import sys
from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from ui import Ui_MainWindow
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Mainwindow,self).__init__()
        self.setupUi(self)
        self.fig = plt.Figure(figsize=(6, 6.5))
        self.canvas = FigureCanvas(self.fig)
        #self.gridlayout = QGridLayout(self.graphicsView)
        self.gridlayout = QGridLayout(self.Learningcurve)
        self.gridlayout.addWidget(self.canvas)
        self.Ax_Y=1
        self.Auto_update.setChecked(True)
        self.Auto_update.toggled.connect(lambda:self.btnstate(self.Auto_update))
        self.Auto_update_judge = True
    def btnstate(self, btn):
        if btn.text()=='自动更新':
            if btn.isChecked()==True:
                self.Auto_update_judge = True
            else:
                self.Auto_update_judge = False
    def plotcos(self, epoch_log,training_MSE_log,val_MSE_log,min_epoch,min_loss):
        ax = self.fig.add_subplot(1,1,1)
        ax.cla()
        #ax.set_xlim(0,100)
        ax.set_ylim(0,self.Ax_Y)
        ax.plot(epoch_log, training_MSE_log)
        ax.plot(epoch_log, val_MSE_log)
        ax.plot(min_epoch,min_loss,'or',alpha=0.5,markersize=6.)
        ax.set_xlabel('Epoch', fontproperties='Times New Roman',visible=True) 
        ax.set_ylabel('MSE', fontproperties='Times New Roman',visible=True) 
        ax.legend(['Training MSE','Cross-validation MSE','Minimum error'],loc = 'upper right')
        self.canvas.draw()
    @pyqtSlot()
    def on_Axmax_clicked(self):
        if self.Ax_Y>=0.1:
            if self.Ax_Y<5:
                self.Ax_Y=self.Ax_Y+0.02
        elif self.Ax_Y>=0.05:
            self.Ax_Y=self.Ax_Y+0.01
        elif self.Ax_Y>=0.01:
            self.Ax_Y=self.Ax_Y+0.005
        else:
            self.Ax_Y=self.Ax_Y+0.001
    @pyqtSlot()
    def on_Axmin_clicked(self):
        if self.Ax_Y>0.1:
            self.Ax_Y=self.Ax_Y-0.02
        elif self.Ax_Y>0.05:
            self.Ax_Y=self.Ax_Y-0.01
        elif self.Ax_Y>0.01:
            self.Ax_Y=self.Ax_Y-0.005
        elif self.Ax_Y>0.001:
            self.Ax_Y=self.Ax_Y-0.001

class Time_Interrupt(Mainwindow):
    def __init__(self):
        super(Time_Interrupt,self).__init__()
        self._timer=QTimer(self)                    # 定时器
        self._timer.timeout.connect(self.Time_out)    # 指向函数
        self._timer.start(27)
        self.update()
        self.epoch=[]
        self.train_loss=[]
        self.total_loss=[]
        self.min_epoch=[]
        self.min_loss=[]
        self.INFO_PRINT=''
    def get_information(self):
        try:
            a = np.loadtxt('loss.txt')
            self.epoch = a[:,0]
            self.train_loss = a[:,1]
            self.total_loss= a[:,2]
            self.min_epoch=a[:,3][-1]
            self.min_loss=a[:,4][-1]
        except:
            pass
        try:
            f = open("INFO.txt","r")
            self.INFO_PRINT = f.read()
            f.close()
        except:
            pass
    def Time_out(self):
        self.plotcos(self.epoch,self.train_loss,self.total_loss,self.min_epoch,self.min_loss)  #画 图
        if self.Auto_update_judge:
            self.get_information()
            self.Trainingdataprint.setText(self.INFO_PRINT)   #文本框逐条添加数据
            self.Trainingdataprint.moveCursor(self.Trainingdataprint.textCursor().End)  #文本框显示到底部

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Time_Interrupt()
    main.show()
    sys.exit(app.exec_())
