import tensorflow as tf
import numpy as np
import cv2
import xlwt
import xlrd
import os
import time
import matplotlib.pyplot as plt
from load import *
from Network import *

class Inference:
    def __init__(self):
        self.ckeckpoint_dir = './checkpoint'    #模型保存地址
        self.test_pic_dir="./test_data/"        #测试数据地址
        self.test_fusion_dir='./fusion/test_data/'
        self.record='./record/'                 #数据记录地址
        self.net=Network()                      #Network类
        self.forward=self.net.VGG_9_            #选择需要的网络
        self.input_data = tf.compat.v1.placeholder(tf.float32, [None,700,500,3], name = "input_data")   #定义输入
        self.supervised_label = tf.compat.v1.placeholder(tf.float32, [None, 1], name = "label")         #定义标签
        self.duckdata=Load_Datasets()           #Load_Datasets类
        self.estimated_weight_max=3.2           #测试区间上限
        self.estimated_weight_min=1.6           #测试区间下限
        self.resultTXT='./result.txt'           #数据预测结果记录
        self.right_area = [str(i)+'.jpg' for i in range(1,9)]       #选择区间
        self.middle_area = [str(i)+'.jpg' for i in range(9,17)]
        self.left_area = [str(i)+'.jpg' for i in range(17,25)]
        self.special_area1 = [str(i)+'.jpg' for i in range(1,5)]
        self.special_area2 = [str(i)+'.jpg' for i in range(5,9)]
        self.special_area3 = [str(i)+'.jpg' for i in range(9,13)]
        self.special_area4 = [str(i)+'.jpg' for i in range(13,17)]
        self.normalize_min = 1.6
        self.normalize_max = 3.2
        self.test_random_fusion_dir='./fusion_random/test_data/'
    def mean(self,a):
        '''计算平均值'''
        return sum(a) / len(a)
    def get_time(self):
        '''获取当前时间'''
        localtime = time.localtime(time.time())
        data_string = str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + '_' + str(localtime[4])
        return data_string
    def mkdir(self,path):
        '''
        创建文件夹
        :param path: 地址
        '''
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)
            print
            path + ' 创建成功'
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print
            path + ' 目录已存在'
            return False
    def del_doc(self,path):
        '''
        删除文件
        :param path: 删除路径
        :return: 无
        '''
        try:
            os.remove(path)
        except:
            pass
    def prepare_init(self):
        '''
        准备工作，计算测试时间，创建存储文件夹，删除不需要的数据
        :return:
        '''
        data_string = self.get_time()
        self.mkdir(self.record+data_string)
        self.del_doc(self.resultTXT)

    def plot_result(self,loss,real,pre):
        '''
        将结果绘制出来
        :param loss: 输入的误差值列表
        :param real: 输入的真实值列表
        :param pre:  输入的预测值列表
        :return:
        '''
        x1 = [0,3.6]
        y1 = [0.05,3.65]
        x2 = [0,3.6]
        y2 = [-0.05,3.55]
        print('平均误差为%g'%(self.mean(loss)))
        x3 = [0,3.6]
        y3 = [0.1,3.7]
        x4 = [0,3.6]
        y4 = [-0.1,3.5]
        plt.plot(real,pre,'o',alpha=0.5)
        plt.plot(x1,y1,'g-')
        plt.plot(x2,y2,'g-')
        plt.plot(x3,y3,'r-')
        plt.plot(x4,y4,'r-')
        plt.xlabel("Original weight(kg)")
        plt.ylabel("Predicted weight(kg)")
        plt.axis([0.8,3.6,0.8,3.6])
        plt.show()

    def evaluate_each_duck_mean_weight(self):
        '''
        预测不同区间内的鸭子质量
        :return:
        '''
        self.prepare_init()
        result = self.forward(self.input_data,is_training=False)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(self.ckeckpoint_dir))
        test_dir = sorted(list(map(int, os.listdir(self.test_pic_dir))))
        pre = []
        real = []
        loss = []
        for duck_num in test_dir:
            label_index, label = self.duckdata.get_label()
            test_data = []
            test_label = []
            img_dir = self.test_pic_dir + str(duck_num) + '/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                img = cv2.imread(img_dir + sec)
                test_data.append(img)
                test_label.append(label[int(np.argwhere(label_index == duck_num))])
            feed_dict = {self.input_data: test_data}
            a = sess.run(result, feed_dict)
            num = a.shape[0]
            a = tf.reshape(a,[num])
            pre1 = self.mean(sess.run(a))
            real1 = test_label[0]
            print((pre1,real1))
            if(real1>=self.estimated_weight_min and real1<self.estimated_weight_max):
                loss.append(abs(real1-pre1))
            pre.append(pre1)
            real.append(real1)
            f = open(self.resultTXT, 'a')
            f.write("%d %g %g %g\n" % (duck_num, pre1, real1, abs(real1-pre1)))
            f.close()
        self.plot_result(loss,real,pre)

    def Interval_judge(self,Interval):
        '''
        判断选取哪个空间
        :param Interval: 可以有多个参数：'L','M','R','S1','S2','S3','S4'代表左边，中间和右边
        :return:
        '''
        if Interval=='L':
            area=self.left_area
        elif Interval=='M':
            area=self.middle_area
        elif Interval=='R':
            area=self.right_area
        elif Interval=='S1':
            area=self.special_area1
        elif Interval=='S2':
            area=self.special_area2
        elif Interval=='S3':
            area=self.special_area3
        elif Interval=='S4':
            area=self.special_area4
        else:
            raise Exception("get_area_data(Interval)输入参数异常，应为'L'或'M'或'R'或'S1'或'S2'或'S3'或'S4'")
        return area

    def evaluate_part_area_duck_mean_weight(self,Interval='M'):
        '''
        根据不同区间预测结果
        :param Interval: 可以有多个参数：'L','M','R','S1','S2','S3','S4'代表左边，中间和右边
        :return:
        '''
        area = self.Interval_judge(Interval)
        self.prepare_init()
        result = self.forward(self.input_data,is_training=False)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(self.ckeckpoint_dir))
        test_dir = sorted(list(map(int, os.listdir(self.test_pic_dir))))
        pre = []
        real = []
        loss = []
        for duck_num in test_dir:
            label_index, label = self.duckdata.get_label()
            test_data = []
            test_label = []
            img_dir = self.test_pic_dir + str(duck_num) + '/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                if sec in area:
                    img = cv2.imread(img_dir + sec)
                    test_data.append(img)
                    test_label.append(label[int(np.argwhere(label_index == duck_num))])
            feed_dict = {self.input_data: test_data}
            a = sess.run(result, feed_dict)
            num = a.shape[0]
            a = tf.reshape(a,[num])
            pre1 = self.mean(sess.run(a))
            real1 = test_label[0]
            # 将预测结果数据打印下来
            print((pre1,real1))
            if(real1>=self.estimated_weight_min and real1<self.estimated_weight_max):
                loss.append(abs(real1-pre1))
            pre.append(pre1)
            real.append(real1)
            f = open(self.resultTXT, 'a')
            f.write("%d %g %g %g\n" % (duck_num, pre1, real1, abs(real1-pre1)))
            f.close()
        self.plot_result(loss,real,pre)
    def evaluate_fusion_image(self):
        '''
        预测不同区间内的鸭子质量
        :return:
        '''
        self.prepare_init()
        result = self.forward(self.input_data,is_training=False)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(self.ckeckpoint_dir))
        test_dir = os.listdir(self.test_fusion_dir)
        pre = []
        real = []
        loss = []
        label_index, labels = self.duckdata.get_label()
        for sec in test_dir:
            test_data = []
            image = cv2.imread(self.test_fusion_dir+sec)
            test_data.append(image)
            feed_dict = {self.input_data: np.array(test_data)}
            a = sess.run(result, feed_dict)
            num = a.shape[0]
            a = tf.reshape(a,[num])
            pre1 = self.mean(sess.run(a))
            real1 = labels[int(sec.split('.')[0])-1]
            print((pre1,real1))
            if(real1>=self.estimated_weight_min and real1<self.estimated_weight_max):
                loss.append(abs(real1-pre1))
            pre.append(pre1)
            real.append(real1)
            f = open(self.resultTXT, 'a')
            f.write("%d %g %g %g\n" % (int(sec.split('.')[0]), pre1, real1, abs(real1-pre1)))
            f.close()
        self.plot_result(loss,real,pre)
    def decoding_weight(self,norm_weight):
        '''
        将归一化的体重计算出来
        :param weight:
        :return: weight
        '''
        weight=norm_weight*(self.normalize_max-self.normalize_min)+self.normalize_min
        return weight

    def evaluate_norm_fusion_image(self):
        '''
        预测不同区间内的正则化鸭子质量
        :return:
        '''
        #文件夹初始化
        self.prepare_init()
        #定义图结构
        result = self.forward(self.input_data,is_training=False)
        #开启会话
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        #加载参数
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(self.ckeckpoint_dir))
        #测试集目录
        test_dir = os.listdir(self.test_fusion_dir)
        pre = []
        real = []
        loss = []
        label_index, labels = self.duckdata.get_label()
        for sec in test_dir:
            test_data = []
            #读取图片数据
            image = cv2.imread(self.test_fusion_dir+sec)
            #获取格式
            test_data.append(image/255.)
            #占位符
            feed_dict = {self.input_data: np.array(test_data)}
            a = sess.run(result, feed_dict)
            num = a.shape[0]
            a = tf.reshape(a,[num])
            #预测值
            pre1 = self.mean(sess.run(a))
            #解码
            pre1 = self.decoding_weight(pre1)
            #真实体重
            real1 = labels[int(sec.split('.')[0])-1]
            # 打印
            print((pre1,real1))
            if(real1>=self.estimated_weight_min and real1<self.estimated_weight_max):
                loss.append(abs(real1-pre1))
            # 预测值
            pre.append(pre1)
            # 真实值
            real.append(real1)
            # 记录
            f = open(self.resultTXT, 'a')
            f.write("%d %g %g %g\n" % (int(sec.split('.')[0]), pre1, real1, abs(real1-pre1)))
            f.close()
        # 可视化结果
        self.plot_result(loss,real,pre)
    def evaluate_random_fusion_mean_weight(self):
        '''
        预测随机融合图像
        :return:
        '''
        self.prepare_init()
        result = self.forward(self.input_data,is_training=False)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(self.ckeckpoint_dir))
        test_dir = sorted(list(map(int, os.listdir(self.test_random_fusion_dir))))
        pre = []
        real = []
        loss = []
        for duck_num in test_dir:
            label_index, label = self.duckdata.get_label()
            img_dir = self.test_random_fusion_dir + str(duck_num) + '/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                test_data = []
                test_label = []
                img = cv2.imread(img_dir + sec)
                test_data.append(img)
                #test_label.append(label[int(np.argwhere(label_index == duck_num))])
                feed_dict = {self.input_data: test_data}
                a = sess.run(result, feed_dict)
                pre1 = a[0][0]
                real1 = label[int(np.argwhere(label_index == duck_num))]
                print((pre1,real1))
                if(real1>=self.estimated_weight_min and real1<self.estimated_weight_max):
                    loss.append(abs(real1-pre1))
                pre.append(pre1)
                real.append(real1)
                f = open(self.resultTXT, 'a')
                f.write("%d %g %g %g\n" % (duck_num, pre1, real1, abs(real1-pre1)))
                f.close()
        self.plot_result(loss,real,pre)


if __name__ == '__main__':
    infer = Inference()
    #infer.evaluate_each_duck_mean_weight()
    infer.evaluate_random_fusion_mean_weight()
