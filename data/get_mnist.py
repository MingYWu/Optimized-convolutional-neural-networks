import os
import gzip
import numpy as np

# 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

def load_data_gz(data_folder):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    # 读取每个文件夹的数据
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 784)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 784)

    return x_train, y_train, x_test, y_test

# # 调用load_data_gz函数加载数据集
# data_folder = '../mnist'
# x_train_gz, y_train_gz, x_test_gz, y_test_gz = load_data_gz(data_folder)
#
# # 查看数据集的形状
# print('x_train_gz:{}'.format(x_train_gz.shape))
# print('y_train_gz:{}'.format(y_train_gz.shape))
# print('x_test_gz:{}'.format(x_test_gz.shape))
# print('y_test_gz:{}'.format(y_test_gz.shape))

def get_mnist_data(num_training=55000, num_validation=5000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data

    data_folder = './dataset/mnist'
    X_train, y_train, X_test, y_test = load_data_gz(data_folder)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    # 打印训练集信息
    print(X_train.shape)  # (60000, 784)

    # Subsample the data
    # 分离数据
    mask = np.arange(num_training, num_training + num_validation)
    X_val = X_train[mask].reshape(num_validation, 28, 28, 1)
    y_val = y_train[mask]

    mask = np.arange(num_training)
    X_train = X_train[mask].reshape(num_training, 28, 28, 1)
    y_train = y_train[mask]

    mask = np.arange(num_test)
    X_test = X_test[mask].reshape(num_test, 28, 28, 1)
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    # 数据标准化
    mean_image = np.mean(X_train)
    std_image = np.std(X_train)

    X_train -= mean_image
    X_train /= std_image
    X_val -= mean_image
    X_val /= std_image
    X_test -= mean_image
    X_test /= std_image

    # Transpose so that channels come first
    # X_train = X_train.transpose(0, 3, 1, 2).copy()
    # X_val = X_val.transpose(0, 3, 1, 2)
    # X_test = X_test.transpose(0, 3, 1, 2)

    # Package data into a dictionary
    # 数据打包成字典
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


