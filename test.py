"""
Author:Mingyang Wu
Day:04.12.2022
Abstract:
Tips:
"""
from data.data_utils import *

import cupy as cp


if __name__ == "__main__":
    data = get_CIFAR10_data(45000, 5000, 10000)

    # batchsize = 256 Val_set_acc : 0.766000, Test_set_acc : 0.751900
    # batchsize = 512 Val_set_acc : 0.764600, Test_set_acc : 0.756200
    # batchsize = 1024, Val_set_acc : 0.773600, Test_set_acc : 0.759300  # adam
    # model_weight_file = "./save_pth/cnn_12_07.pkl"

    # cnn_add_2
    # batchsize=256, Val_set_acc : 0.817600, Test_set_acc : 0.808900
    # model_weight_file = "./save_pth/cnn_add_8176.pkl" 
    # batchsize=256, Val_set_acc : 0.817400, Test_set_acc : 0.805000
    # model_weight_file = "./save_pth/cnn_add_8174.pkl"


    # cnn_12_07 -> best
    # batch_size=256, Val_set_acc : 0.826000, Test_set_acc : 0.816500
    # batch_size=128, Val_set_acc : 0.820200, Test_set_acc : 0.811800
    model_weight_file = "./save_pth/cnn_12_07_best_82.pkl"  
    
    # batchsize=256, Val_set_acc : 0.766000, Test_set_acc : 0.751900
    # model_weight_file = "./save_pth/cnn_12_07.pkl"
    
    # batchsize=256, Val_set_acc : 0.820400, Test_set_acc : 0.816200
    # batchsize=128, Val_set_acc : 0.821600, Test_set_acc : 0.811400
    # batchsize=64 , Val_set_acc : 0.808200, Test_set_acc : 0.800300
    # model_weight_file = "./save_pth/cnn_12_07_best_816.pkl"   
    
    # batchsize=128, Val_set_acc : 0.819800, Test_set_acc : 0.809000
    # batchsize=256, Val_set_acc : 0.822400, Test_set_acc : 0.814400
    # model_weight_file = "./save_pth/cnn_12_07_8198.pkl"  
    
    # vgg_14_bn
    # batchsize=128, Val_set_acc : 0.718600, Test_set_acc : 0.709900
    # batchsize=256, Val_set_acc : 0.728600, Test_set_acc : 0.714900
    # model_weight_file = "./save_pth/model_vgg14_bn_71.pkl"
    
    # batchsize=128, Val_set_acc : 0.718600, Test_set_acc : 0.709900
    # model_weight_file='./save_pth/cifar10/vgg_14_bn.pkl'

    # 选择模型的权重，其中这个模型要存在 lib/model/ 文件夹下
    model = load_models(model_weight_file)
    batch_size = 256

    X_val = cp.asarray(data['X_val']).transpose(0, 3, 1, 2)
    y_val = cp.asarray(data['y_val'])
    N_val = data['y_val'].shape[0]

    X_test = cp.asarray(data['X_test']).transpose(0, 3, 1, 2)
    y_test = cp.asarray(data['y_test'])
    N_test = data['X_test'].shape[0]

    acc = 0
    val_acc = 0
    for i in range(0, N_val, batch_size):
        # start = i*batch_size
        end = min(i + batch_size, N_val)
        acc += sum(cp.argmax(model.loss(X_val[i:end]), axis=1) == y_val[i:end])

    val_acc = acc / N_val

    acc = 0
    test_acc = 0
    for i in range(0, N_test, batch_size):
    # start = i*batch_size
        end = min(i + batch_size, N_test)
        acc += sum(cp.argmax(model.loss(X_test[i:end]), axis=1) == y_test[i:end])

    test_acc = acc / N_test

    print("cifar10: 203吴明洋: 1200205120, 203李想: 1200113220, 204黄玄亨: 1200570048")
    print("Val_set_acc : %f, Test_set_acc : %f" %(val_acc, test_acc))

    # vgg_13->bug
    # model_weight_file ='./save_pth/cifar10/vgg13.pkl'
    
    # vgg_14->bug
    # model_weight_file='./save_pth/cifar10/vgg14_629.pkl'
    # model_weight_file='./save_pth/cifar10/vgg14_55.pkl'
    

