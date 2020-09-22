import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
# REGULARIZER = 0.01
BATCH_SIZE = 10

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict = {xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict = {xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape, weight_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = weight_name)

def bias_variable(shape, bias_name):
    initial = tf.constant(1.5, shape = shape)
    return tf.Variable(initial, name = bias_name)

def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] =1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")  # padding="SAME"用零填充边界，VALID是不填充

def conv2d_2(x, W):
    # stride[1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="VALID")  # padding="SAME"用零填充边界，VALID是不填充

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

def pic_init(pic):
    image = cv2.imread(pic)
    image = cv2.resize(image, (80, 253))
    b, g, r = cv2.split(image)
    thresh, img2 = cv2.threshold(r, 30, 0, cv2.THRESH_TOZERO)
    cv2.imwrite('out.jpg',img2)
    predict_data = []
    predict_data.append(img2)
    predict_data = np.array(predict_data)
    predict_data = predict_data.astype(np.float32)
    return predict_data

def backward(pic):
    Test_image = pic_init(pic)
    X = tf.placeholder(tf.float32, [None, 253, 80], name = "X")
    Y_ = tf.placeholder(tf.float32, [None, 1])
    LEARNING_RATE_BASE = 0.00001  # 最初学习率
    LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
    LEARNING_RATE_STEP = 1000  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE
    gloabl_steps = tf.Variable(0, trainable=False)  # 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, gloabl_steps,LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    x_image = tf.reshape(X, [-1, 253, 80, 1])
    W_conv1 = weight_variable([3, 3, 1, 4], weight_name="conv1")  # kernel 3*3, channel is 3
    b_conv1 = bias_variable([4], bias_name="cb1")
    h_conv1 = conv2d(x_image, W_conv1) + b_conv1
    h_conv1_ac = tf.nn.softplus(h_conv1)  # output size 251*78*32
    h_pool1 = avg_pool_2x2(h_conv1_ac)  # output size 125*39*32

    ## conv2 layer ##
    W_conv2 = weight_variable([3, 3, 4, 8], weight_name="conv2")  # kernel 3*3, in size 3, out size 5
    b_conv2 = bias_variable([8], bias_name="cb2")
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_conv2_ac = tf.nn.softplus(h_conv2)  # output size 123*37*5
    h_pool2 = avg_pool_2x2(h_conv2_ac)  # output size 61*18*16

    ## conv3 layer ##
    W_conv3 = weight_variable([3, 3, 8, 16], weight_name="conv3")  # kernel 3*3, in size 3, out size 5
    b_conv3 = bias_variable([16], bias_name="cb3")
    h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3
    h_conv3_ac = tf.nn.softplus(h_conv3)  # output size 59*16*16
    h_pool3 = avg_pool_2x2(h_conv3_ac)  # output size 29*8*16

    ## funcl layer ##
    W_fc1 = weight_variable([29 * 8 * 16, 32], weight_name="fc1")
    b_fc1 = bias_variable([32], bias_name="fb1")

    # [n_samples,7,7,64]->>[n_samples, 7*7*64]
    h_pool3_flat = tf.reshape(h_pool3, [-1, 29 * 8 * 16])
    h_fc1 = tf.nn.softplus(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## func2 layer ##
    W_fc2 = weight_variable([32, 16], weight_name="fc2")
    b_fc2 = bias_variable([16], bias_name="fb2")

    h_fc2 = tf.nn.softplus(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    ## func3 layer ##
    W_fc3 = weight_variable([16, 1], weight_name="fc3")
    b_fc3 = bias_variable([1], bias_name="fb3")
    fc3 = tf.matmul(h_fc2_drop, W_fc3)

    y = tf.add(fc3, b_fc3, name="y")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint('./checkpoint/')
        saver.restore(sess, model_file)
        graph = tf.get_default_graph()
        # 卷积
        conv1_4 = sess.run(y, feed_dict={X: Test_image, keep_prob: 1.0})  # [1, 28, 28 ,16]
        conv3_transpose = sess.run(tf.transpose(conv1_4, [1, 0]))
        fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(1, 1))
        ax3.imshow(conv3_transpose)
        ax3.set_xticks([])
        ax3.set_yticks([])
        # for j in range(1):
        #     ax3[j].imshow(conv3_transpose[j][0])  # tensor的切片[row, column]
        #     ax3[j].set_xticks([])
        #     ax3[j].set_yticks([])
        # # 池化
        # conv3_16 = sess.run(h_pool3, feed_dict={X: Test_image, keep_prob: 1.0})  # [1, 28, 28 ,16]
        # conv3_transpose = sess.run(tf.transpose(conv3_16, [3, 0, 1, 2]))
        # fig3, ax3 = plt.subplots(nrows=2, ncols=8, figsize=(8, 2))
        # for i in range(2):
        #     for j in range(8):
        #         ax3[i][j].imshow(conv3_transpose[i*4+j][0])  # tensor的切片[row, column]
        #         ax3[i][j].set_xticks([])
        #         ax3[i][j].set_yticks([])
        #plt.title('Pool3 16x29x8')

        # plt.savefig("./pic/Pool3.png")
        plt.show()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    img_dir = input("请输入图片地址：")
    backward(img_dir)

if __name__ == '__main__':
    main()
