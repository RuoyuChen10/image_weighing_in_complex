# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'learningcurve.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(960, 902)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Learningcurve = QtWidgets.QGroupBox(self.centralwidget)
        self.Learningcurve.setGeometry(QtCore.QRect(40, 40, 751, 431))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.Learningcurve.setFont(font)
        self.Learningcurve.setObjectName("Learningcurve")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 470, 71, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.Trainingdataprint = QtWidgets.QTextBrowser(self.centralwidget)
        self.Trainingdataprint.setGeometry(QtCore.QRect(40, 500, 891, 341))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.Trainingdataprint.setFont(font)
        self.Trainingdataprint.setObjectName("Trainingdataprint")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(800, 150, 131, 141))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Axmax = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(16)
        self.Axmax.setFont(font)
        self.Axmax.setObjectName("Axmax")
        self.verticalLayout.addWidget(self.Axmax)
        self.Axmin = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(16)
        self.Axmin.setFont(font)
        self.Axmin.setObjectName("Axmin")
        self.verticalLayout.addWidget(self.Axmin)
        self.Auto_update = QtWidgets.QRadioButton(self.centralwidget)
        self.Auto_update.setGeometry(QtCore.QRect(200, 470, 131, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(16)
        self.Auto_update.setFont(font)
        self.Auto_update.setObjectName("Auto_update")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 10, 211, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(20)
        self.label_2.setFont(font)
        self.label_2.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 960, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Learningcurve.setTitle(_translate("MainWindow", "Learning Curves"))
        self.label.setText(_translate("MainWindow", "工作区"))
        self.Axmax.setText(_translate("MainWindow", "增大Y轴"))
        self.Axmin.setText(_translate("MainWindow", "减小Y轴"))
        self.Auto_update.setText(_translate("MainWindow", "自动更新"))
        self.label_2.setText(_translate("MainWindow", "训练监控上位机"))
