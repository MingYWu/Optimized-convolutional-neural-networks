import matplotlib.pyplot as plt
import cupy as cp
import datetime
import os

# from data.data_utils import get_CIFAR10_data
from data.get_mnist import get_mnist_data
from solver import Solver

from lib.model_mnist.cnn import *


if __name__ == '__main__':
    # 获取cifar10里面的数据
    # 训练集， 验证集， 测试集
    # data = get_CIFAR10_data(45000, 5000, 10000)
    data = get_mnist_data()
    batch_size=1024

    # 获得模型
    model = ThreeLayerConvNet(reg=0.008)

    solver = Solver(model, data,
                    num_epochs=60,
                    batch_size=batch_size,
                    print_every=5, 
                    update_rule='adam',
                    # update_rule='sgd_momentum',
                    lr_decay=0.91,
                    lr_decay_epochs = 3,
                    optim_config={'learning_rate': 0.0001, 'momentum': 0.9})
    solver.train()

    best_model = model
    # 数据转化
    X_val = cp.asarray(data['X_val']).transpose(0, 3, 1, 2)
    y_val = cp.asarray(data['y_val'])
    X_test = cp.asarray(data['X_test']).transpose(0, 3, 1, 2)
    y_test = cp.asarray(data['y_test'])

    # 最后一次验证集准确度测试
    val_size = data['X_val'].shape[0]
    test_size = data['X_test'].shape[0]
    acc=0
    for i in range(0, val_size, batch_size):
        end = min(i + batch_size, val_size)
        acc += sum(cp.argmax(best_model.loss(X_val[i:end]), axis=1) == y_val[i:end])
    print('Validation set accuracy: ', acc / val_size)

    # 最后一次测试集准确度测试
    acc = 0
    val_size = X_val.shape[0]
    for i in range(0, test_size, batch_size):
        end = min(i + batch_size, test_size)
        acc += sum(cp.argmax(best_model.loss(X_test[i:end]), axis=1) == y_test[i:end])
    print('Test set accuracy: ', acc / test_size)