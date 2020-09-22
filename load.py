import cv2
import numpy as np
import xlrd
import random
import os

class Load_Datasets:
    '''
    涉及数据预处理的一些函数
    '''
    def __init__(self):
        self.weight_label='./weight.txt'
        self.training_data_dir='./training_data/'
        self.test_data_dir='./test_data/'
        self.right_area    = [str(i)+'.jpg' for i in range(1,9)]
        self.middle_area   = [str(i)+'.jpg' for i in range(9,17)]
        self.left_area     = [str(i)+'.jpg' for i in range(17,25)]
        self.special_area1 = [str(i)+'.jpg' for i in range(1,5)]
        self.special_area2 = [str(i)+'.jpg' for i in range(5,9)]
        self.special_area3 = [str(i)+'.jpg' for i in range(9,13)]
        self.special_area4 = [str(i)+'.jpg' for i in range(13,17)]
    def get_label(self):
        '''获取标签'''
        data = np.loadtxt(self.weight_label)
        label = data[:,1]
        label_index = data[:,0]
        return label_index, label
    def Sequential_disruption(self,train_data, train_label, test_data, test_label):
        '''此函数对训练集与数据集进行随机的打乱，防止其整齐'''
        train_num = len(train_label)
        test_num = len(test_label)
        train_seq_distruption = [i for i in range(0, train_num)]
        test_seq_distruption = [i for i in range(0, test_num)]
        random.shuffle(train_seq_distruption)
        random.shuffle(test_seq_distruption)
        distrupted_train_data = []
        distrupted_train_label = []
        distrupted_test_data = []
        distrupted_test_label = []
        for i in train_seq_distruption:
            distrupted_train_data.append(train_data[i])
            distrupted_train_label.append(train_label[i])
        for i in test_seq_distruption:
            distrupted_test_data.append(test_data[i])
            distrupted_test_label.append(test_label[i])
        return np.array(distrupted_train_data), np.array(distrupted_train_label), np.array(distrupted_test_data), np.array(distrupted_test_label)
    def Sequential_disruption_data(self,train_data, train_label):
        '''此函数对训练集与数据集进行随机的打乱，防止其整齐'''
        train_num = len(train_label)
        train_seq_distruption = [i for i in range(0, train_num)]
        random.shuffle(train_seq_distruption)
        distrupted_train_data = []
        distrupted_train_label = []
        for i in train_seq_distruption:
            distrupted_train_data.append(train_data[i])
            distrupted_train_label.append(train_label[i])
        return distrupted_train_data, distrupted_train_label

    def get_data(self):
        train_dir = sorted(list(map(int, os.listdir(self.training_data_dir))))
        test_dir = sorted(list(map(int, os.listdir(self.test_data_dir))))
        label_index, label = self.get_label()
        training_data = []
        training_label = []
        test_data = []
        test_label = []
        for duck_num in train_dir:
            img_dir = self.training_data_dir+str(duck_num)+'/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                img = cv2.imread(img_dir+sec)
                if img is None:
                        print('Error')
                training_data.append(img)
                training_label.append([label[int(np.argwhere(label_index==duck_num))]])
            print(duck_num)
        for duck_num in test_dir:
            img_dir = self.test_data_dir+str(duck_num)+'/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                img = cv2.imread(img_dir+sec)
                if img is None:
                        print('Error')
                test_data.append(img)
                test_label.append([label[int(np.argwhere(label_index==duck_num))]])
            print(duck_num)
        training_data,training_label,test_data,test_label = self.Sequential_disruption(training_data,training_label,test_data,test_label)
        return training_data,training_label,test_data,test_label
    def get_train_datasets(self):
        test_dir = sorted(list(map(int, os.listdir(self.training_data_dir))))
        label_index, label = get_label()
        test_data = []
        test_label = []
        test_num = []
        for duck_num in test_dir:
            img_dir = self.training_data_dir + str(duck_num) + '/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                img = cv2.imread(img_dir + sec)
                if img is None:
                    print('Error')
                test_data.append(img)
                test_label.append(label[int(np.argwhere(label_index == duck_num))])
                test_num.append(str(duck_num))
            print(duck_num)
        #test_data, test_label = Sequential_disruption(test_data, test_label)
        return test_data, test_label,test_num

    def get_test_datasets(self):
        test_dir = sorted(list(map(int, os.listdir(self.test_data_dir))))
        label_index, label = get_label()
        test_data = []
        test_label = []
        test_num = []
        for duck_num in test_dir:
            img_dir = self.test_data_dir + str(duck_num) + '/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                img = cv2.imread(img_dir + sec)
                test_data.append(img)
                test_label.append(label[int(np.argwhere(label_index == duck_num))])
                test_num.append(str(duck_num))
            print(duck_num)
        #test_data, test_label = Sequential_disruption(test_data, test_label)
        return test_data, test_label,test_num
    def Interval_judge(self,Interval):
        '''判断选取哪个空间'''
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
    def get_area_data(self,Interval='M'):
        '''
        :param Interval: Interval可以有三个参数：'L','M','R','S1','S2','S3','S4'代表左边，中间和右边
        :return: 对应区间的图像
        '''
        area = self.Interval_judge(Interval)
        train_dir = sorted(list(map(int, os.listdir(self.training_data_dir))))
        test_dir = sorted(list(map(int, os.listdir(self.test_data_dir))))
        label_index, label = self.get_label()
        training_data = []
        training_label = []
        test_data = []
        test_label = []
        for duck_num in train_dir:
            img_dir = self.training_data_dir+str(duck_num)+'/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                if sec in area:
                    img = cv2.imread(img_dir+sec)
                    if img is None:
                            print('Error')
                    training_data.append(img)
                    training_label.append([label[int(np.argwhere(label_index==duck_num))]])
            print(duck_num)
        for duck_num in test_dir:
            img_dir = self.test_data_dir+str(duck_num)+'/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                if sec in area:
                    img = cv2.imread(img_dir+sec)
                    if img is None:
                            print('Error')
                    test_data.append(img)
                    test_label.append([label[int(np.argwhere(label_index==duck_num))]])
            print(duck_num)
        training_data,training_label,test_data,test_label = self.Sequential_disruption(training_data,training_label,test_data,test_label)
        return training_data,training_label,test_data,test_label
