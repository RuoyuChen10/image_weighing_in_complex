import tensorflow as tf
import numpy as np    
import os
from load import Load_Datasets
from Fusion import Fusion
from Network import *
import matplotlib.pyplot as plt
import math
import shutil

class Train_API:
    '''
    Using for training data.
    '''
    def __init__(self):
        super(Train_API,self).__init__()
        self.net=multi_scale_net()
        self.forward=self.net.MSN     #选择需要的网络
        self.duckdata=None#Fusion()          #初始化类
        self.datasets=None
        self.label =None
        self.test_data =None
        self.test_label = None#self.duckdata.get_random_fusion_data()#读取数据
        self.GPU_num = 2                #使用显卡数量
        self.mini_batch = 10            #batch size=GPU_num*mini_batch
        self.STEPS = 10000000           #最大步数
        self.LEARNING_RATE_BASE = 0.00001  # 最初学习率
        self.LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
        self.LEARNING_RATE_STEP = 1000  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/mini_batch
        self.global_step =tf.compat.v1.train.get_or_create_global_step() #步数
        self.input_data = tf.compat.v1.placeholder(tf.float32, [None,700,500,3], name = "input_data")#定义输入
        self.supervised_label = tf.compat.v1.placeholder(tf.float32, [None, 1], name = "label")#定义标签
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.LEARNING_RATE_BASE, self.global_step, self.LEARNING_RATE_STEP, self.LEARNING_RATE_DECAY, staircase=True)#学习率衰减
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)#优化器选择
        self.cuda_max_memory=0.9    #显卡最大显存控制
        self.min_epoch = 0          #记录最低点步数
        self.min_loss = 1           #记录最低点loss
        self.print_INFO='Start training!'
        self.Print_update=True
        self.INFO_dir='./INFO.txt'
        self.loss_dir='./loss.txt'
        self.logs_dir="./logs"
        self.ckpt_dir='./checkpoint'
        self.ckpt_save_dir='./checkpoint/variable'
    def mkdir(self,name):
        '''创建文件夹'''
        isExists=os.path.exists(name)
        if not isExists:
            os.makedirs(name)
        return 0
    def del_dirs(self,path):
        '''删除文件夹'''
        try:
            shutil.rmtree(path)
        except:
            pass
    def del_doc(self,path):
        '''删除文件'''
        try:
            os.remove(path)
        except:
            pass
    def del_log(self):
        self.del_dirs(self.logs_dir)    #递归删除文件夹
        self.del_doc(self.INFO_dir)
        self.del_doc(self.loss_dir)

    def average_gradients(self,tower_grads):
        '''平均梯度，这个是多gpu时的计算'''
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, axis=0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    def compute_loss(self,test_data,test_label, loss_mse, sess, X, Y, cp_num=200):
        '''分布计算loss，这是因为电脑服务器不足以一次性运算，每次计算500个'''
        loss = 0
        num = len(test_data)
        if num!=0:
            for j in range(0, math.ceil(num / cp_num)):
                if (j != math.ceil(num / cp_num) - 1):
                    loss = loss + sess.run(loss_mse, feed_dict={X: test_data[cp_num*j : cp_num*j+cp_num], Y: test_label[cp_num*j : cp_num*j+cp_num]})*cp_num
                else:
                    loss = loss + sess.run(loss_mse, feed_dict={X: test_data[cp_num * j:num], Y: test_label[cp_num * j:num]})*(num - cp_num * j)
            loss = loss / num
        return loss
    def multi_gpu_apply(self):
        '''
        设置多显卡操作
        '''
        #总梯度
        tower_grads = []
        #总loss
        tower_loss = []
        #多GPU运算设置
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),reuse=tf.compat.v1.AUTO_REUSE):
            for i in range(self.GPU_num):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        #一块显卡批量输入
                        _x = self.input_data[i * self.mini_batch:(i + 1) * self.mini_batch]
                        _y = self.supervised_label[i * self.mini_batch:(i + 1) * self.mini_batch]
                        #计算前向传播
                        logits = self.forward(_x, is_training=True)
                        #定义损失函数
                        loss_mse = tf.reduce_mean(tf.square(logits-_y))
                        #计算梯度
                        grads = self.opt.compute_gradients(loss_mse)
                        #记录这一块显卡loss
                        tower_loss.append(loss_mse)
                        #计算这一块显卡梯度
                        tower_grads.append(grads)
                        #定义测试的图结构（非训练状态）
                        if i==0:
                            logits_test = self.forward(_x, is_training=False)
                            loss_mse_test = tf.reduce_mean(tf.square(logits_test-_y))
        #计算平均损失
        mean_loss = tf.stack(axis=0, values=tower_loss)
        mean_loss = tf.reduce_mean(mean_loss, 0)
        #优化方法
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = self.average_gradients(tower_grads)
            train_op = self.opt.apply_gradients(grads, global_step=self.global_step)
        return train_op,mean_loss,loss_mse_test
    def cuda_memory_setting(self):
        '''GPU设置'''
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = self.cuda_max_memory
        tf_config.gpu_options.allow_growth = True # 自适应显存
        return tf_config
    def save_variable_list(self):
        '''保存多余变量'''
        var_list = tf.compat.v1.trainable_variables()
        if self.global_step is not None:
            var_list.append(self.global_step)
        g_list = tf.compat.v1.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        return var_list
    def save_training_information(self,epoch, train_loss, total_loss):
        '''保存训练时loss等日志数据'''
        f = open(self.loss_dir, 'a')
        f.write("%g %g %g %g %g\n" % (epoch, train_loss, total_loss,self.min_epoch, self.min_loss))
        f.close()
        self.print_INFO="After %g epoch, loss_mse on train data is %g, loss_mse on val data is %g, at %g epoch get min_loss %g\n" % (epoch, train_loss, total_loss,self.min_epoch, self.min_loss)
        self.Print_update=True
        print(self.print_INFO)
        f = open(self.INFO_dir, 'a')
        f.write(self.print_INFO)
        f.close()
    def save_minimum_variable(self,sess,total_loss,epoch,i,saver):
        '''保存参数'''
        if total_loss < self.min_loss:
            self.min_loss = total_loss
            self.min_epoch = epoch
            saver.save(sess, self.ckpt_save_dir, global_step=i)
    def train(self):
        '''训练'''
        #删除之前的日志文件
        self.del_log()
        #创建文件夹
        self.mkdir(self.ckpt_dir)
        #设置多进程
        with tf.device("/cpu:0"):
            #设置多显卡操作
            train_op,mean_loss,loss_mse_test=self.multi_gpu_apply()
            #记录数据到tensorboard中
            merged = tf.compat.v1.summary.merge_all()
            #GPU显存设置
            #tf_config=self.cuda_memory_setting()
            tf_config = tf.ConfigProto(allow_soft_placement=True)
            #with tf.compat.v1.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            with tf.compat.v1.Session(config=tf_config) as sess:
                #变量初始化
                with tf.name_scope('init'):
                    init_op = tf.compat.v1.global_variables_initializer()
                sess.run(init_op)
                var_list = self.save_variable_list()
                #保存模型变量定义
                saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=5)
                #开始记录图结构
                #writer = tf.summary.FileWriter("./logs", sess.graph)
                writer = tf.compat.v1.summary.FileWriter(self.logs_dir, sess.graph)
                #读取是否需要加载之前的模型文件
                if tf.train.latest_checkpoint(self.ckpt_dir) is not None:
                    saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_dir))
                #一些变量的存放
                data_len = len(self.datasets)
                # 训练模型。
                for i in range(self.STEPS):
                    #计算epoch
                    epoch = i / int(len(self.datasets)/self.mini_batch/self.GPU_num)
                    start = (i*self.mini_batch*self.GPU_num) % int(data_len/(self.mini_batch*self.GPU_num))*self.mini_batch*self.GPU_num
                    end = start + self.mini_batch*self.GPU_num
                    #计算当前损失率
                    if i % 5 == 0:  # 相当于epoch=1
                        train_loss = sess.run(loss_mse_test, feed_dict={self.input_data: self.datasets[start:end], self.supervised_label: self.label[start:end]})
                        #total_loss = sess.run(loss_mse_test, feed_dict={self.input_data: self.test_data, self.supervised_label: self.test_label})
                        total_loss = self.compute_loss(self.test_data,self.test_label, loss_mse_test, sess, self.input_data, self.supervised_label)
                        self.save_minimum_variable(sess,total_loss,epoch,i,saver)
                        self.save_training_information(epoch, train_loss, total_loss)
                    #if epoch % 5 == 0:
                        #self.datasets, self.label = self.duckdata.Sequential_disruption_data(self.datasets, self.label)
                    #优化参数
                    sess.run(train_op, feed_dict={self.input_data: self.datasets[start:end], self.supervised_label: self.label[start:end]})

