from train import Train_API
from load import Load_Datasets

'''
训练单图输入模式
'''

if __name__ == '__main__':
    main =  Train_API()
    main.duckdata=Load_Datasets()
    main.datasets, main.label, main.test_data, main.test_label = main.duckdata.get_data()#读取数据
    main.train()
