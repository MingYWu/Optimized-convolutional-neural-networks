from lib.utils.layers import *


def conv_relu_forward(x, weight, bias, conv_param):
    # 卷积+非线性激活的正向传播
    out_feature, conv_cache = conv_forward(x, weight, bias, conv_param)
    out, relu_cache = relu_forward(out_feature)
    cache = (conv_cache, relu_cache)

    return out, cache


def conv_relu_backward(dout, cache):
    # 卷积+非线性激活的反向传播
     conv_cache, relu_cache = cache
     da = relu_backward(dout, relu_cache)
     dx, dw, db = conv_backward(da, conv_cache)

     return dx, dw, db



def conv_pool_forward(x, w, b, conv_param, pool_param):
    # 卷积池化正向传播
    a, conv_cache = conv_forward(x, w, b, conv_param)
    out, pool_cache = max_pool_forward(a, pool_param)
    cache = (conv_cache, pool_cache)
    return out, cache


def conv_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, pool_cache = cache
    ds = max_pool_backward(dout, pool_cache)
    dx, dw, db = conv_backward(ds, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, conv_param, bn_param):
    """
        conv-batchnorm-relu
    """
    # 矩阵运算卷积
    a, conv_cache = conv_forward(x, w, b, conv_param)
    N = a.shape[0]
    # reshape变化
    a_ = a.reshape(N, -1)
    gamma = bn_param.get('gamma')
    beta = bn_param.get('beta')
    # 批归一化  卷积之后大小变化
    a_bn, cache_bn = batchnorm_forward(a_, gamma, beta, bn_param)
    a = a_bn.reshape(a.shape)
    # 非线性激活的正向传播
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache, cache_bn)

    return out, cache


def conv_bn_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, bn_cache = cache
    # relu 反向传播
    da = relu_backward(dout, relu_cache)
    # 图像变化
    N = da.shape[0]
    # 获得参数dgamma dbeta参数
    da_bn, dgamma, dbeta = batchnorm_backward(da.reshape(N, -1), bn_cache)
    da = da_bn.reshape(da.shape)
    # 矩阵反向求导
    dx, dw, db = conv_backward(da, conv_cache)

    return dx, dw, db, dgamma, dbeta


def conv_bn_forward(x, w, b, conv_param, bn_param):
    # 卷积层+批量归一化
    a, conv_cache = conv_forward(x, w, b, conv_param)

    N = a.shape[0]
    a_ = a.reshape(N, -1)
    # 
    gamma = bn_param.get('gamma')
    beta = bn_param.get('beta')
    a_bn, cache_bn = batchnorm_forward(a_, gamma, beta, bn_param)

    out = a_bn.reshape(a.shape)
    cache = (conv_param, bn_param)

    return out, cache


def conv_bn_backward(dout, cache):
    conv_cache, bn_cache = cache

    N = dout.shape[0]
    da_bn, dgamma, dbeta = batchnorm_backward(dout.reshape(N, -1), bn_cache)
    da = da_bn.reshape(dout.shape)
    dx, dw, db = conv_backward(da, conv_cache)

    return dx, dw, db, dgamma, dbeta


# def conv_res_relu_backward(dout, cache):
#      conv_cache, relu_cache = cache
#      da = relu_backward(dout, relu_cache)
#      dx, dw, db = conv_res_backward(da, conv_cache)

#      return dx, dw, db