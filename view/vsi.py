import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
# REGULARIZER = 0.01
BATCH_SIZE = 10

def pic_init(pic):
    image = cv2.imread(pic)
    predict_data = []
    predict_data.append(image)
    predict_data = np.array(predict_data)
    return predict_data

def cov_blog(net,is_training=False,name=None):
    '''
    定义的卷积层
    '''
    net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN1'+name)
    ## convl layer ##192*192
    net1 = tf.contrib.layers.conv2d(net,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv1'+name)
    net2 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool1'+name)
    net = tf.contrib.layers.batch_norm(net2, is_training=is_training, scope='BN2'+name)
    # ## conv2 layer ##
    net3 = tf.contrib.layers.conv2d(net,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv2'+name)
    net4 = tf.contrib.layers.max_pool2d(net3,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool2'+name)
    net = tf.contrib.layers.batch_norm(net4, is_training=is_training, scope='BN3'+name)
    # ## conv3 layer ##
    net5 = tf.contrib.layers.conv2d(net,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv3'+name)
    net6 = tf.contrib.layers.max_pool2d(net5,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool3'+name)
    net = tf.contrib.layers.batch_norm(net6, is_training=is_training, scope='BN4'+name)
    # ## conv4 layer ##
    net7 = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4'+name)
    net8 = tf.contrib.layers.conv2d(net7,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv5'+name)
    net9 = tf.contrib.layers.max_pool2d(net8,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool4'+name)
    net = tf.contrib.layers.batch_norm(net9, is_training=is_training, scope='BN5'+name)
    # ## conv5 layer ##
    net10 = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv6'+name)
    net11 = tf.contrib.layers.max_pool2d(net10,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool5'+name)
    net = tf.contrib.layers.flatten(net11, scope='flatten'+name)
    return net,net10

def MSN(input_net,is_training=False):
    net1 = tf.contrib.layers.avg_pool2d(input_net,kernel_size=(1,1),stride=(1,1),padding='valid', scope='Pool1')
    net2 = tf.contrib.layers.avg_pool2d(input_net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Pool2')
    net3 = tf.contrib.layers.avg_pool2d(input_net,kernel_size=(4,4),stride=(4,4),padding='valid', scope='Pool3')

    net1,_ = cov_blog(net1,is_training=is_training,name='_1')
    net2,_ = cov_blog(net2,is_training=is_training,name='_2')
    net3,ob = cov_blog(net3,is_training=is_training,name='_3')

    net = tf.concat([net1,net2,net3],1,name = 'concat')

    ## funcl layer ##
    net = tf.contrib.layers.fully_connected(net,num_outputs=512,activation_fn=tf.nn.softplus, scope='fully_connected1')
    #net = tf.contrib.layers.dropout(net,keep_prob=keep_prob,is_training=is_training, scope='dropout1')
    ## func2 layer ##
    net = tf.contrib.layers.fully_connected(net,num_outputs=256,activation_fn=tf.nn.softplus, scope='fully_connected2')
    #net = tf.contrib.layers.dropout(net,keep_prob=keep_prob,is_training=is_training, scope='dropout2')
    ## func3 layer ##
    net = tf.contrib.layers.fully_connected(net,num_outputs=1,activation_fn=None, scope='logits')
    return net,ob

def backward(pic):
    Test_image = pic_init(pic)
    X = tf.compat.v1.placeholder(tf.float32, [None,700,500,3], name = "input_data")
    Y_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name = "label")#定义标签
    is_training=False

    net,net1 = MSN(X,is_training=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint('./checkpoint/')
        saver.restore(sess, model_file)
        graph = tf.get_default_graph()
        # 池化
        conv3_16 = sess.run(net1, feed_dict={X: Test_image})  # [1, 28, 28 ,16]
        conv3_transpose = sess.run(tf.transpose(conv3_16, [3, 0, 1, 2]))
        nrows = 8
        ncols = 8
        fig3, ax3 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols,nrows))
        for i in range(nrows):
            for j in range(ncols):
                ax3[i][j].imshow(conv3_transpose[j+i*ncols][0])  # tensor的切片[row, column]
                ax3[i][j].set_xticks([])
                ax3[i][j].set_yticks([])
        #plt.title('Pool3 16x29x8')




        # conv3_transpose = sess.run(tf.transpose(conv3_16, [1,0]))
        # fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(1,1))
        #
        # ax3.imshow(conv3_transpose)  # tensor的切片[row, column]
        # # ax3[0][0].set_xticks([])
        # # ax3[0][0].set_yticks([])

        plt.savefig("./Pool3.png")
        plt.show()

def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    img_dir = input("请输入图片地址：")
    backward(img_dir)

if __name__ == '__main__':
    main()
