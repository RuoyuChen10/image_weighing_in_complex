# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random
from load import Load_Datasets

class Fusion():
    def __init__(self):
        super(Fusion,self).__init__()
        self.original_train_datasets_dir = 'training_data/'
        self.original_test_datasets_dir = 'test_data/'
        self.save_root_path = './fusion/'
        self.weight_label='./weight.txt'
        self.train_datasets_dir = './fusion/training_data/'
        self.test_datasets_dir = './fusion/test_data/'
        self.save_fusion_root_path = './fusion_random/'
        self.random_training_data_dir='./fusion_random/training_data/'
        self.random_test_data_dir='./fusion_random/test_data/'
        self.random_fusion_num = 6  #用于随机融合图像数量
        self.ramdom_fusion_generate_num = 30 #用于生成随机融合图像的数量
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
    def get_label(self):
        '''
        获取标签
        '''
        data = np.loadtxt(self.weight_label)
        label = data[:,1]
        label_index = data[:,0]
        return label_index.tolist(), label.tolist()
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
        return np.array(distrupted_train_data), np.array(distrupted_train_label)
    def Generate(self):
        '''
        用于生成融合图像
        For generate fusion image as datasets
        -----------------------------------------
        原始图像目录
        Original image catalog example:
        root|----training_data|----1|----1.jpg
            |                 |     |----2.jpg
            |                 |      ...
            |                 |     |----m.jpg
            |                 |----2|----1.jpg
            |                 |     |----2.jpg
            |                 |      ...
            |                 |     |----m.jpg
            |                 ...
            |                 |----n|----1.jpg
            |                      |----2.jpg
            |                      ...
            |                      |----m.jpg
            |----test_data|----4|----1.jpg
                          |     |----2.jpg
                          |     ...
                          |     |----m.jpg
                          |----5|----1.jpg
                          |     |----2.jpg
                          |     ...
                          |     |----m.jpg
                          ...
                          |----n|----1.jpg
                                |----2.jpg
                                ...
                                |----m.jpg
        '''

        for dirs_ in [self.original_train_datasets_dir,self.original_test_datasets_dir]:
            #读取文件夹内目录
            dirs = os.listdir(dirs_)
            for dir_ in dirs:
                #获取内部图片目录
                image_dirs = os.listdir(dirs_+dir_)
                #融合图像计数
                image_num = 0
                for image_dir in image_dirs:
                    #读取图片
                    img = cv2.imread(dirs_+dir_+'/'+image_dir)
                    #转换类型，防止超过255会求余数不累积
                    img = np.array(img,dtype='int')
                    if image_num == 0:
                        #定义第一个融合图像
                        fusion_img = np.copy(img)
                    else:
                        #图像融合相加
                        fusion_img = fusion_img+img
                    #融合图像计数
                    image_num+=1
                #将数值更新为正常图片模式下
                fusion_img =fusion_img/image_num
                #fusion_img.astype('uint8')
                #新建文件夹
                self.mkdir(self.save_root_path+dirs_)
                #保持图像
                cv2.imwrite(self.save_root_path+dirs_+dir_+'.jpg',fusion_img)
        print('------------------Done------------------')
    def Generate_random_fusion_image(self):
        '''
        用于生成随机融合图像
        For generate fusion image as datasets
        -----------------------------------------
        原始图像目录
        Original image catalog example:
        root|----training_data|----1|----1.jpg
            |                 |     |----2.jpg
            |                 |      ...
            |                 |     |----m.jpg
            |                 |----2|----1.jpg
            |                 |     |----2.jpg
            |                 |      ...
            |                 |     |----m.jpg
            |                 ...
            |                 |----n|----1.jpg
            |                      |----2.jpg
            |                      ...
            |                      |----m.jpg
            |----test_data|----4|----1.jpg
                          |     |----2.jpg
                          |     ...
                          |     |----m.jpg
                          |----5|----1.jpg
                          |     |----2.jpg
                          |     ...
                          |     |----m.jpg
                          ...
                          |----n|----1.jpg
                                |----2.jpg
                                ...
                                |----m.jpg
        '''

        for dirs_ in [self.original_train_datasets_dir,self.original_test_datasets_dir]:
            #读取文件夹内目录
            dirs = os.listdir(dirs_)
            for dir_ in dirs:
                #获取内部图片目录
                image_dirs = os.listdir(dirs_+dir_)
                for i in range(1,self.ramdom_fusion_generate_num+1):
                    choose_dirs = random.sample(image_dirs,self.random_fusion_num)
                    image_num = 0
                    for image_dir in choose_dirs:
                        #读取图片
                        img = cv2.imread(dirs_+dir_+'/'+image_dir)
                        #转换类型，防止超过255会求余数不累积
                        img = np.array(img,dtype='int')
                        if image_num == 0:
                            #定义第一个融合图像
                            fusion_img = np.copy(img)
                            image_num = 1
                        else:
                            #图像融合相加
                            fusion_img = fusion_img+img
                    fusion_img =fusion_img/self.random_fusion_num
                    #新建文件夹
                    self.mkdir(self.save_fusion_root_path+dirs_+dir_)
                    #保持图像
                    cv2.imwrite(self.save_fusion_root_path+dirs_+'/'+dir_+'/'+str(i)+'.jpg',fusion_img)
        print('------------------Done------------------')
    def load_fusion_image(self,path):
        '''
        输入数据集路径，返回数据集格式
        :param path: 数据集存放路径
        '''
        label_index, labels=self.get_label()
        data = []
        label= []
        image_dirs = os.listdir(path)
        for image_dir in image_dirs:
            label.append([labels[int(image_dir.split('.')[0])-1]])
            image = cv2.imread(path+image_dir)
            data.append(image)
        return np.array(data),np.array(label)
    def load_fusion_datasets(self):
        '''
        得到训练集，测试集
        '''
        train_data,train_label=self.load_fusion_image(self.train_datasets_dir)
        test_data,test_label=self.load_fusion_image(self.test_datasets_dir)
        return train_data,train_label,test_data,test_label
    def get_random_fusion_data(self):
        '''
        获取随机融合的数据集
        :return:
        '''
        train_dir = sorted(list(map(int, os.listdir(self.random_training_data_dir))))
        test_dir = sorted(list(map(int, os.listdir(self.random_test_data_dir))))
        label_index, label = self.get_label()
        training_data = []
        training_label = []
        test_data = []
        test_label = []
        for duck_num in train_dir:
            img_dir = self.random_training_data_dir+str(duck_num)+'/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                img = cv2.imread(img_dir+sec)
                if img is None:
                        print('Error')
                training_data.append(img)
                training_label.append([label[duck_num-1]])
            print(duck_num)
        for duck_num in test_dir:
            img_dir = self.random_test_data_dir+str(duck_num)+'/'
            time_dir = os.listdir(img_dir)
            for sec in time_dir:
                img = cv2.imread(img_dir+sec)
                if img is None:
                        print('Error')
                test_data.append(img)
                test_label.append([label[duck_num-1]])
            print(duck_num)
        training_data,training_label,test_data,test_label = self.Sequential_disruption(training_data,training_label,test_data,test_label)
        return np.array(training_data),np.array(training_label),np.array(test_data),np.array(test_label)
