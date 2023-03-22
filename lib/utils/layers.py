from .gpu import *
import numpy as np
import cupy as cp

def conv_forward(x, weight, bias, conv_param):
    # kernal_size = 3, stride = 1, padding = 1
    # 获得参数
    stride, padding = conv_param["stride"], conv_param['pad']
    # 创建填充
    x_pad = cp.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
    # 利用矩阵计算宽，加快计算
    out, col = conv(x_pad, weight, bias, stride)
    # 存储卷积层中的参数
    cache = (x, weight, bias, stride, padding, col)

    return out, cache

def conv_backward(dout, cache):
    """
        卷积的反向求导，matrix
        forward 的反向操作
    """
    # 从cache中获取卷积层中的参数
    x, weight, bias, stride, padding, col = cache
    KN, Kchannel, KH, KW = weight.shape

    # pad 填充数组input_data
    if KH == 1:
        dout_padded = dout
    else:
        dout_padded = cp.pad(dout,
                             [(0, 0), (0, 0), (KH - stride - 1, KH - stride - 1), (KW - stride - 1, KW - stride - 1)])
    # 变形
    rw = weight[:, :, ::-1, ::-1]
    rw = rw.transpose([1, 0, 2, 3])

    # 矩阵运算
    dx, colt = conv_ba(dout_padded, rw, stride)
    ecol = dout.transpose([0, 2, 3, 1]).reshape([-1, KN])
    dw = ecol.T @ col
    dw = dw.reshape([KN, Kchannel, KH, KW])

    # 计算dout关于bias的导数
    db = cp.sum(dout, axis=(0, 2, 3))
    return dx, dw, db

def max_pool_forward(x, pool_param):
    # 获得输入参数的数据，池化核的大小
    pool_h, pool_w = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.shape
    # 计算经过池化特征提取后输出特征图的大小
    out_h = int(1 + (H - pool_h) / stride)  # (1 + (16-3)/3)
    out_w = int(1 + (W - pool_w) / stride)

    # 其实是运用im2col算法将图像数组 展开
    col = image_to_column(x, pool_h, pool_w, stride)
    col = col.reshape(-1, pool_h * pool_w)

    # 获得每一行中 最大值的那个数据
    arg_max = cp.argmax(col, axis=1)
    out = cp.max(col, axis=1)

    # 先输入特征图变化成多维的，再转换
    out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    # self.x = x
    # self.arg_max = arg_max
    # 存储图像数据信息
    cache = (x, arg_max, pool_h, pool_w, stride)

    return out, cache

def max_pool_backward(dout, cache):
    # 最大池化的反向传播，获取池化层的数据信息
    x, arg_max, pool_h, pool_w, stride = cache

    dout = dout.transpose(0, 2, 3, 1)

    pool_size = pool_h * pool_w
    dmax = cp.zeros((dout.size, pool_size))
    dmax[cp.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (pool_size,))

    dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
    dx = column_to_image(dcol, x.shape, pool_h, pool_w, stride)

    return dx

def ReLU(x):
    """ReLU non-linearity."""
    return cp.maximum(0, x)

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = ReLU(x)
    cache = x

    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    # relu的反向传播
    dx, x = None, cache
    dx = dout
    # 对于x小于等于0的值赋值为0
    dx[x <= 0] = 0

    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    # 批量归一化的正向传播 优化了梯度流动
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', cp.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', cp.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == 'train':
        # 求均值
        sample_mean = cp.mean(x, axis=0)  # cp.mean([[1,2],[3,4]])->[2,3]
        # 求方差
        sample_var = cp.var(x, axis=0)
        # 训练过程中使用每个Batch的均值和方差做归一化，预测过程中则通过训练数据进行估算
        out_ = (x - sample_mean) / cp.sqrt(sample_var + eps)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        out = gamma * out_ + beta
        cache = (out_, x, sample_var, sample_mean, eps, gamma, beta)
    elif mode == 'test':
        scale = gamma / cp.sqrt(running_var + eps)
        out = x * scale + (beta - running_mean * scale)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None

    out_, x, sample_var, sample_mean, eps, gamma, beta = cache

    N = x.shape[0]
    dout_ = gamma * dout
    dvar = cp.sum(dout_ * (x - sample_mean) * -0.5 * (sample_var + eps) ** -1.5, axis=0)
    dx_ = 1 / cp.sqrt(sample_var + eps)
    dvar_ = 2 * (x - sample_mean) / N

    # intermediate for convenient calculation
    di = dout_ * dx_ + dvar * dvar_
    dmean = -1 * cp.sum(di, axis=0)
    dmean_ = cp.ones_like(x) / N

    dx = di + dmean * dmean_
    dgamma = cp.sum(dout * out_, axis=0)
    dbeta = cp.sum(dout, axis=0)

    return dx, dgamma, dbeta

def affine_forward(x, w, b):
    # 全连接层
    out = None
    # Reshape x into rows
    N = x.shape[0]
    x_row = x.reshape(N, -1)  # (N,D)
    out = cp.dot(x_row, w) + b  # (N,M)
    cache = (x, w, b)

    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache

    dx, dw, db = None, None, None
    dx = cp.dot(dout, w.T)  # (N,D)
    dx = cp.reshape(dx, x.shape)  # (N,d1,...,d_k)
    x_row = x.reshape(x.shape[0], -1)  # (N,D)
    dw = cp.dot(x_row.T, dout)  # (D,M)
    db = cp.sum(dout, axis=0, keepdims=True)  # (1,M)

    return dx, dw, db

def dropout_forward(x, level):
    retain_prob = 1. - level
    # 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
    # 硬币 正面的概率为p，n表示每个神经元试验的次数
    # 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。

    # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    sample = cp.random.binomial(n=1, p=retain_prob, size=x.shape)
    # 0、1与x相乘，我们就可以屏蔽某些神经元，让它们的值变为0
    x *= sample
    x /= retain_prob

    return x

# def conv_res_backward(dout, cache):
#     """
#         卷积的反向求导，matrix
#         forward 的反向操作
#     """
#     # 从cache中获取卷积层中的参数
#     x, weight, bias, stride, padding, col = cache
#     KN, Kchannel, KH, KW = weight.shape

#     # pad 填充数组input_data
#     # print(KH -stride - 1, KH - stride - 1)
#     if KH == 1:
#         dout_padded = cp.pad(dout,
#                              [(0, 0), (0, 0), (1, 1), (1, 1)])
#     else:
#         dout_padded = cp.pad(dout, [(0, 0), (0, 0), (KH - stride - 1, KH - stride - 1), (KW - stride - 1, KW - stride - 1)])
#     # 卷积核心反转
#     rw = weight[:, :, ::-1, ::-1]
#     rw = rw + cp.ones(rw.shape)
#     rw = rw.transpose([1, 0, 2, 3])

#     # dout 关于x的求导
#     dx, colt = conv(dout_padded, rw, bias,stride)
#     # 误差转化为矩阵
#     ecol = dout.transpose([0, 2, 3, 1]).reshape([-1, KN])

#     # 计算dout关于w的导数
#     dw = ecol.T @ col
#     dw = dw.reshape([KN, Kchannel, KH, KW])

#     # 计算dout关于bias的导数
#     db = cp.sum(dout, axis=(0, 2, 3))
#     return dx, dw, db