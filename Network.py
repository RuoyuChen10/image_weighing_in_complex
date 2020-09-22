import tensorflow as tf

class Network:
    def __init__(self):
        self.keep_prob=0.8
        self.dropout_rate=0.2
    def VGG_9_(self,network_input,is_training=True):
        net = tf.contrib.layers.batch_norm(network_input, is_training=is_training, scope='BN1')
        ## convl layer ##192*192
        net = tf.contrib.layers.conv2d(net,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv1')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool1')
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN2')
        # ## conv2 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv2')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool2')
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN3')
        # ## conv3 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv3')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool3')
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN4')
        # ## conv4 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4')
        net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv5')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool4')
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN5')
        # ## conv5 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid', activation_fn=tf.nn.softplus, scope='Conv6')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool5')
        net = tf.contrib.layers.flatten(net, scope='flatten')
        ## funcl layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=256,activation_fn=tf.nn.softplus, scope='fully_connected1')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout1')
        ## func2 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=128,activation_fn=tf.nn.softplus, scope='fully_connected2')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout2')
        ## func3 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=1,activation_fn=None, scope='logits')
        return net

    def VGG_9(self,network_input,is_training=True):
        # conv layer 1
        net = tf.keras.layers.BatchNormalization(name='BN1')(network_input,training=is_training)
        net = tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), strides=(1, 1), padding='valid',activation=tf.nn.softplus,data_format='channels_last',name='Conv1')(net)
        net = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid',data_format='channels_last',name='MaxPool1')(net)

        # conv layer 2
        net = tf.keras.layers.BatchNormalization(name='BN2')(net,training=is_training)
        net = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1, 1), padding='valid',activation=tf.nn.softplus,data_format='channels_last',name='Conv2')(net)
        net = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid',data_format='channels_last',name='MaxPool2')(net)

        # conv layer 3
        net = tf.keras.layers.BatchNormalization(name='BN3')(net,training=is_training)
        net = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1, 1), padding='valid',activation=tf.nn.softplus,data_format='channels_last',name='Conv3')(net)
        net = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid',data_format='channels_last',name='MaxPool3')(net)

        # conv layer 4
        net = tf.keras.layers.BatchNormalization(name='BN4')(net,training=is_training)
        net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='valid',activation=tf.nn.softplus,data_format='channels_last',name='Conv4')(net)
        net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='valid',activation=tf.nn.softplus,data_format='channels_last',name='Conv5')(net)
        net = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid',data_format='channels_last',name='MaxPool4')(net)

        # conv layer 5
        net = tf.keras.layers.BatchNormalization(name='BN5')(net,training=is_training)
        net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='valid',activation=tf.nn.softplus,data_format='channels_last',name='Conv6')(net)
        net = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid',data_format='channels_last',name='MaxPool5')(net)

        #Fallten
        net = tf.keras.layers.Flatten(data_format='channels_last',name='Flatten')(net)

        #FC1
        net = tf.keras.layers.Dense(units=256,activation=tf.nn.softplus,name='FC1')(net)
        net = tf.keras.layers.Dropout(rate=self.dropout_rate,name='Dropout1')(net,training=is_training)

        #FC2
        net = tf.keras.layers.Dense(units=128,activation=tf.nn.softplus,name='FC2')(net)
        net = tf.keras.layers.Dropout(rate=self.dropout_rate,name='Dropout2')(net,training=is_training)

        #OUT
        net = tf.keras.layers.Dense(units=1,activation=None,name='Output')(net)
        return net

    def VGG_9_norm(self,network_input,is_training=True):
        net = tf.contrib.layers.batch_norm(network_input, is_training=is_training, scope='BN1')
        ## convl layer ##192*192
        net = tf.contrib.layers.conv2d(net,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv1')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool1')
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN2')
        # ## conv2 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv2')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool2')
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN3')
        # ## conv3 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv3')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool3')
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN4')
        # ## conv4 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4')
        net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv5')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool4')
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN5')
        # ## conv5 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv6')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool5')
        net = tf.contrib.layers.flatten(net, scope='flatten')
        ## funcl layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=256,activation_fn=tf.nn.softplus, scope='fully_connected1')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout1')
        ## func2 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=128,activation_fn=tf.nn.softplus, scope='fully_connected2')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout2')
        ## func3 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=1,activation_fn=tf.nn.sigmoid, scope='logits')
        return net

    def VGG_9_pyramind(self,network_input,is_training=True):
        ## convl layer ##192*192
        net1 = tf.contrib.layers.batch_norm(network_input, is_training=is_training, scope='BN1')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv1')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool1')

        # ## conv2 layer ##
        net1 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN2')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv2')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool2')

        # ## conv3 layer ##
        net1 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN3')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv3')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool3')

        net2 =tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN4_2')
        net2 = tf.contrib.layers.conv2d(net2,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4_2')
        net2 = tf.contrib.layers.flatten(net2, scope='flatten2')

        # ## conv4 layer ##
        net1 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN4')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv5')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool4')

        net3 =tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN4_3')
        net3 = tf.contrib.layers.conv2d(net3,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4_3')
        net3 = tf.contrib.layers.flatten(net3, scope='flatten3')

        # ## conv5 layer ##
        net1 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN5')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv6')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool5')

        net0 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN6')
        net0 = tf.contrib.layers.conv2d(net0,num_outputs=128,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv7')
        net0 = tf.contrib.layers.max_pool2d(net0,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool6')
        net0 = tf.contrib.layers.flatten(net0, scope='flatten0')

        net1 = tf.contrib.layers.flatten(net1, scope='flatten1')
        net = tf.concat([net0,net1,net2,net3],1,name = 'concat')

        ## funcl layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=512,activation_fn=tf.nn.softplus, scope='fully_connected1')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout1')
        ## func2 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=256,activation_fn=tf.nn.softplus, scope='fully_connected2')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout2')
        ## func3 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=1,activation_fn=None, scope='logits')
        return net
    def VGG_9_pyramind_2(self,network_input,is_training=True):
        ## convl layer ##192*192
        net1 = tf.contrib.layers.batch_norm(network_input, is_training=is_training, scope='BN1')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv1')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool1')

        # ## conv2 layer ##
        net1 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN2')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv2')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool2')

        # ## conv3 layer ##
        net1 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN3')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv3')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool3')

        # net2 =tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN4_2')
        # net2 = tf.contrib.layers.conv2d(net2,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4_2')
        net2 = tf.contrib.layers.flatten(net1, scope='flatten2')

        # ## conv4 layer ##
        net1 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN4')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv5')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool4')

        # net3 =tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN4_3')
        # net3 = tf.contrib.layers.conv2d(net3,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4_3')
        net3 = tf.contrib.layers.flatten(net1, scope='flatten3')

        # ## conv5 layer ##
        net1 = tf.contrib.layers.batch_norm(net1, is_training=is_training, scope='BN5')
        net1 = tf.contrib.layers.conv2d(net1,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv6')
        net1 = tf.contrib.layers.max_pool2d(net1,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool5')

        net1 = tf.contrib.layers.flatten(net1, scope='flatten1')
        net = tf.concat([net1,net2,net3],1,name = 'concat')

        ## funcl layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=512,activation_fn=tf.nn.softplus, scope='fully_connected1')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout1')
        ## func2 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=256,activation_fn=tf.nn.softplus, scope='fully_connected2')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout2')
        ## func3 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=1,activation_fn=None, scope='logits')
        return net

class multi_scale_net():
    def __init__(self):
        self.keep_prob=0.8
        self.dropout_rate=0.2
    def cov_blog(self,net,is_training=True,name=None):
        '''
        定义的卷积层
        '''
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN1'+name)
        ## convl layer ##192*192
        net = tf.contrib.layers.conv2d(net,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv1'+name)
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool1'+name)
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN2'+name)
        # ## conv2 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv2'+name)
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool2'+name)
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN3'+name)
        # ## conv3 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv3'+name)
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool3'+name)
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN4'+name)
        # ## conv4 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv4'+name)
        net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv5'+name)
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool4'+name)
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN5'+name)
        # ## conv5 layer ##
        net = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.softplus, scope='Conv6'+name)
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool5'+name)
        net = tf.contrib.layers.flatten(net, scope='flatten'+name)
        return net

    def MSN(self,input_net,is_training=True):
        net1 = tf.contrib.layers.avg_pool2d(input_net,kernel_size=(1,1),stride=(1,1),padding='valid', scope='Pool1')
        net2 = tf.contrib.layers.avg_pool2d(input_net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Pool2')
        net3 = tf.contrib.layers.avg_pool2d(input_net,kernel_size=(4,4),stride=(4,4),padding='valid', scope='Pool3')

        net1 = self.cov_blog(net1,is_training=is_training,name='_1')
        net2 = self.cov_blog(net2,is_training=is_training,name='_2')
        net3 = self.cov_blog(net3,is_training=is_training,name='_3')

        net = tf.concat([net1,net2,net3],1,name = 'concat')

        ## funcl layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=512,activation_fn=tf.nn.softplus, scope='fully_connected1')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout1')
        ## func2 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=256,activation_fn=tf.nn.softplus, scope='fully_connected2')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout2')
        ## func3 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=1,activation_fn=None, scope='logits')
        return net
