import cupy as cp


def conv(x, weight, bias, stride):
    """
        using matrix to calculate conv layer
        image to col
        conv_param = stride, padding
    """
    # 转换为数组类型
    img = cp.asarray(x)
    weight = cp.asarray(weight)

    # 获得batchsize大小，输入通道数，高，宽
    N, in_channel, H, W = img.shape
    # 获得输出的通道数，输入的通道数，卷积核的大小
    out_channel, in_channel, kernal_H, kernal_W = weight.shape

    # 输出特征图大小, img已经是padding后的shape。34 - 3 + 1
    H_new = int(1 + (H - kernal_H) / stride)
    W_new = int(1 + (W - kernal_W) / stride)

    # 索引序列, 基于输入特征图和卷积核大小，移动步长去建立一个索引
    idxh = cp.arange(kernal_H)[:, cp.newaxis] + cp.zeros([in_channel, 1, kernal_W])
    idxw = cp.arange(kernal_W)[cp.newaxis, :] + cp.zeros([in_channel, kernal_H, 1])
    idxc = cp.arange(in_channel)[:, cp.newaxis, cp.newaxis] + cp.zeros([1, kernal_H, kernal_W])

    # 需要去创建索引值
    idxh = idxh.reshape([1, -1]) + (cp.arange(H_new)[:, cp.newaxis] + cp.zeros([W_new])).reshape([-1, 1]) * stride
    idxw = idxw.reshape([1, -1]) + (cp.arange(W_new) + cp.zeros([H_new, 1])).reshape([-1, 1]) * stride
    idxc = idxc.reshape([1, -1]) + (cp.zeros([H_new, W_new])).reshape([-1, 1])

    idxh = cp.int32(idxh.get())
    idxw = cp.int32(idxw.get())
    idxc = cp.int32(idxc.get())
    # 行，一行代表感受野位置
    # 在输入数据上，根据卷积核大小，将三个通道一次展开为一维数组，然后再连接为一个长的一位数组，
    # 再根据步幅，将输入数据中每个应用卷积核的地方都会生成一个一维数组
    col = cp.asarray(img[:, idxc, idxh, idxw])

    # 卷积运算
    # 将卷积核 纵向转化成一个列向量
    kernal_list = cp.asarray(weight.reshape([out_channel, -1]).T)
    # feature_map = col @ kernal_list  # *
    # 进行一个卷积运算
    feature_map = col @ kernal_list + bias    # *  输出数据是二维的，因此需要reshape
    out = feature_map.reshape([N, H_new, W_new, out_channel]).transpose([0, 3, 1, 2])

    return out, col.reshape([-1, in_channel*kernal_H*kernal_W])

def conv_ba(x, weight, stride):
    """
        using matrix to calculate conv layer
        image to col
        conv_param = stride, padding
    """
    img = cp.asarray(x)
    weight = cp.asarray(weight)

    N, in_channel, H, W = img.shape
    out_channel, in_channel, kernal_H, kernal_W = weight.shape

    H_new = int(1 + (H - kernal_H) / stride)
    W_new = int(1 + (W - kernal_W) / stride)
    
    idxh = cp.arange(kernal_H)[:, cp.newaxis] + cp.zeros([in_channel, 1, kernal_W])
    idxw = cp.arange(kernal_W)[cp.newaxis, :] + cp.zeros([in_channel, kernal_H, 1])
    idxc = cp.arange(in_channel)[:, cp.newaxis, cp.newaxis] + cp.zeros([1, kernal_H, kernal_W])

    # 创建全图的索引，idxh，idxw，idxc组成一个索引坐标。
    idxh = idxh.reshape([1, -1]) + (cp.arange(H_new)[:, cp.newaxis] + cp.zeros([W_new])).reshape([-1, 1]) * stride
    idxw = idxw.reshape([1, -1]) + (cp.arange(W_new) + cp.zeros([H_new, 1])).reshape([-1, 1]) * stride
    idxc = idxc.reshape([1, -1]) + (cp.zeros([H_new, W_new])).reshape([-1, 1])

    idxh = cp.int32(idxh.get())
    idxw = cp.int32(idxw.get())
    idxc = cp.int32(idxc.get())
    # 行，一行代表感受野位置
    # 在输入数据上，根据卷积核大小，将三个通道一次展开为一维数组，然后再连接为一个长的一位数组，
    # 再根据步幅，将输入数据中每个应用卷积核的地方都会生成一个一维数组
    col = cp.asarray(img[:, idxc, idxh, idxw])

    # 卷积运算
    kernal_list = cp.asarray(weight.reshape([out_channel, -1]).T)
    feature_map = col @ kernal_list
    out = feature_map.reshape([N, H_new, W_new, out_channel]).transpose([0, 3, 1, 2])

    return out, col.reshape([-1, in_channel*kernal_H*kernal_W])

def image_to_column(input_data, filter_h, filter_w, stride=1, pad=0):
    """_summary_
        将图像，根据卷积核去转化为一列一列的，
        跟conv算法其实是相似的
    Args:
        input_data (_type_): _description_
        filter_h (_type_): _description_
        filter_w (_type_): _description_
        stride (int, optional): _description_. Defaults to 1.
        pad (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # 输入数据的数据量，通道数，长宽
    N, C, H, W = input_data.shape
    # 计算输出特征图的大小
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # 重建填充，注意pad是0还是1
    img = cp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # 创建一个大小和输出特征图一致的全0的数组
    col = cp.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 变化图像序列
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def column_to_image(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """_summary_

    Args:
        col (_type_): _description_
        input_shape (_type_): _description_
        filter_h (_type_): _description_
        filter_w (_type_): _description_
        stride (int, optional): _description_. Defaults to 1.
        pad (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = cp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def conv_bias(x, weight, bias, stride, padding):
    """
        using matrix to calculate conv layer
        image to col
        conv_param = stride, padding
        添加了一定的偏置项，其实在缓解过拟合上起到的效果甚小
    """
    img = cp.asarray(x)
    weight = cp.asarray(weight)

    N, in_channel, H, W = img.shape
    out_channel, in_channel, kernal_H, kernal_W = weight.shape

    # 输出特征图大小
    H_new = int(1 + (H + 2 * padding - kernal_H) // stride)
    W_new = int(1 + (W + 2 * padding - kernal_W) // stride)

    # 创建单个tensor size，索引序列, 卷积核
    idxh = cp.arange(kernal_H)[:, cp.newaxis] + cp.zeros([in_channel, 1, kernal_W])
    idxw = cp.arange(kernal_W)[cp.newaxis, :] + cp.zeros([in_channel, kernal_H, 1])
    idxc = cp.arange(in_channel)[:, cp.newaxis, cp.newaxis] + cp.zeros([1, kernal_H, kernal_W])

    # 创建全图的索引，idxh，idxw，idxc组成一个索引坐标
    idxh = idxh.reshape([1, -1]) + (cp.arange(H_new)[:, cp.newaxis] + cp.zeros([W_new])).reshape([-1, 1]) * stride
    idxw = idxw.reshape([1, -1]) + (cp.arange(W_new) + cp.zeros([H_new, 1])).reshape([-1, 1]) * stride
    idxc = idxc.reshape([1, -1]) + (cp.zeros([H_new, W_new])).reshape([-1, 1])

    idxh = cp.int32(idxh.get())
    idxw = cp.int32(idxw.get())
    idxc = cp.int32(idxc.get())
    # 行，一行代表感受野位置
    col = cp.asarray(img[:, idxc, idxh, idxw])

    # 卷积运算
    kernal_list = cp.asarray(weight.reshape([out_channel, -1]).T)
    feature_map = col @ kernal_list + bias     # *
    out = feature_map.reshape([N, H_new, W_new, out_channel]).transpose([0, 3, 1, 2])

    return out, col.reshape([-1, in_channel*kernal_H*kernal_W])