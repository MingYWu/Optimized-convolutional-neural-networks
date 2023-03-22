import pickle as pickle
import os
import numpy as np
import cupy as cp

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    # 打开数据集
    with open(filename, 'rb') as f:
        # 将文件中的数据解析为一个Python对象
        datadict = pickle.load(f,encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        # 获取一个batch的数据
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        X = np.array(X)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        # 根据数据集名称，装载数据集
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        # 加载一个batch的数据
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    # 加载test数据集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=500, num_validation=50, num_test=50):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data

    # 数据集路径
    cifar10_dir = r'./dataset/cifar-10-batches-py'
    # 获得数据
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # 改变元素的数据类型
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print(X_train.shape)  # 打印数据集个数
    # Subsample the data 切割样本集
    mask = np.arange(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = np.arange(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    # 测试集
    mask = np.arange(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image  样本数据标准化
    mean_image = np.mean(X_train, axis=0)
    std_image = np.std(X_train, axis=0)

    X_train -= mean_image
    X_train /= std_image
    X_val -= mean_image
    X_val /= std_image
    X_test -= mean_image
    X_test /= std_image
    
    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def load_models(model_file):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt) will
    be skipped.

    Inputs:
    - models_dir: String giving the path to a directory containing model_cifar10 files.
    Each model_cifar10 file is a pickled dictionary with a 'model_cifar10' field.

    Returns:
    A dictionary mapping model_cifar10 file names to models.
    """
    # 加载模型的权重
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model
