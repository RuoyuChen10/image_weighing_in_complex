from train import Train_API
from Fusion import Fusion
from Network import *

'''
融合单图输入模式
'''

if __name__ == '__main__':
    main =  Train_API()
    main.net=multi_scale_net()
    main.forward=main.net.MSN     #选择需要的网络
    main.duckdata=Fusion()
    main.datasets, main.label, main.test_data, main.test_label = main.duckdata.get_random_fusion_data()#读取数据

    main.train()
