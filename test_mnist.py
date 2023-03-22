from data.data_utils import *
import cupy as cp
from data.get_mnist import *
# from lib.model_mnist.cnn_test import *

if __name__ == "__main__":
    data = get_mnist_data()
    
    # batch_size = 256, Val_set_acc : 0.992000, Test_set_acc : 0.990600
    # batch_size = 512, Val_set_acc : 0.992600, Test_set_acc : 0.991800
    # batch_size = 1024, Val_set_acc : 0.994000, Test_set_acc : 0.992000
    # model_weight_file = "./save_pth/mnist/cnn_mnist.pkl"
    
    # Val_set_acc : 0.994000, Test_set_acc : 0.993000
    model_weight_file = './save_pth/mnist/cnn_993.pkl'

    # 选择模型的权重，其中这个模型要存在lib/model_mnist/文件夹下
    model = load_models(model_weight_file)
    # 调整batchsize
    batch_size = 256

    X_val = cp.asarray(data['X_val']).transpose(0, 3, 1, 2)
    y_val = cp.asarray(data['y_val'])
    N_val = data['y_val'].shape[0]

    X_test = cp.asarray(data['X_test']).transpose(0, 3, 1, 2)
    y_test = cp.asarray(data['y_test'])
    N_test = data['X_test'].shape[0]

    # 验证集准确度测试
    acc = 0
    val_acc = 0
    for i in range(0, N_val, batch_size):
        # start = i*batch_size
        end = min(i + batch_size, N_val)
        acc += sum(cp.argmax(model.loss(X_val[i:end]), axis=1) == y_val[i:end])

    val_acc = acc / N_val

    # 测试集准确度测试
    acc = 0
    test_acc = 0
    for i in range(0, N_test, batch_size):
    # start = i*batch_size
        end = min(i + batch_size, N_test)
        acc += sum(cp.argmax(model.loss(X_test[i:end]), axis=1) == y_test[i:end])
    test_acc = acc / N_test

    print("手写数据集:203吴明洋:1200205120, 203李想: 1200113220, 204黄玄亨: 1200570048")
    print("Val_set_acc : %f, Test_set_acc : %f" %(val_acc, test_acc))