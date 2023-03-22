from lib.utils.modules import *
from lib.utils.layers import *
from lib.utils.loss import *

class ThreeLayerConvNet(object):
    """
        在cnn_add_2的基础上添加一些卷积层
    """

    def __init__(self, input_dim=(3, 32, 32), weight_scale=1, reg=0.0, dtype=cp.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize weights and biases
        C, H, W = input_dim
        # 32 * 32
        self.params['W1_1'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (3 * 32 * 32)) ** 0.5,
                                                              size=(64, 3, 3, 3))
        self.params['W1_2'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (64 * 32 * 32)) ** 0.5,
                                                              size=(64, 64, 3, 3))  # (N, D)
        # self.params['W1_3'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (128 * 32 * 32)) ** 0.5,
        #                                                       size=(128, 128, 3, 3))  # (N, D)
        self.params['b1_1'] = cp.ones(64)
        self.params['b1_2'] = cp.ones(64)
        self.params['beta1_1'] = cp.zeros(65536)  # 存在独立的维度参数  (D,1)  # 32*32*64
        self.params['gamma1_1'] = cp.ones(65536)
        self.params['beta1_2'] = cp.zeros(65536)
        self.params['gamma1_2'] = cp.ones(65536)
        # self.params['beta1_3'] = cp.zeros(65536*2)
        # self.params['gamma1_3'] = cp.ones(65536*2)

        # 16 * 16
        self.params['W2_1'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (128 * 16 * 16)) ** 0.5,
                                                              size=(128, 64, 3, 3))
        self.params['W2_2'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (128 * 16 * 16)) ** 0.5,
                                                              size=(128, 128, 3, 3))
        # self.params['W2_3'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (236 * 16 * 16)) ** 0.5,
        #                                                       size=(256, 256, 3, 3))
        self.params['b2_1'] = cp.ones(128)
        self.params['b2_2'] = cp.ones(128)
        self.params['beta2_1'] = cp.zeros(32768)  # 存在独立的维度参数  (D,1)  # 16*16*128
        self.params['gamma2_1'] = cp.ones(32768)
        self.params['beta2_2'] = cp.zeros(32768)
        self.params['gamma2_2'] = cp.ones(32768)
        # self.params['beta2_3'] = cp.zeros(32768*2)
        # self.params['gamma2_3'] = cp.ones(32768*2)

        # 8 * 8
        self.params['W3_1'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (128 * 8 * 8)) ** 0.5,
                                                              size=(256, 128, 3, 3))
        self.params['W3_2'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (256 * 8 * 8)) ** 0.5,
                                                              size=(256, 256, 3, 3))
        self.params['b3_1'] = cp.ones(256)
        self.params['b3_2'] = cp.ones(256)
        self.params['beta3_1'] = cp.zeros(16384)  # 存在独立的维度参数  (D,1)
        self.params['gamma3_1'] = cp.ones(16384)
        self.params['beta3_2'] = cp.zeros(16384)  # 8*8*512
        self.params['gamma3_2'] = cp.ones(16384)


        # 4 * 4  concat
        self.params['W4_1'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (256 * 4 * 4)) ** 0.5,
                                                              size=(512, 256, 3, 3))
        self.params['W4_2'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (512 * 4 * 4)) ** 0.5,
                                                              size=(512, 512, 3, 3))
        self.params['b4_1'] = cp.ones(512)
        self.params['b4_2'] = cp.ones(512)
        self.params['beta4_1'] = cp.zeros(8192)  # 存在独立的维度参数  (D,1)  # 4*4*512
        self.params['gamma4_1'] = cp.ones(8192)
        self.params['beta4_2'] = cp.zeros(8192)
        self.params['gamma4_2'] = cp.ones(8192)

        # 2 * 2
        # self.params['W5_1'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (512 * 2 * 2)) ** 0.5,
        #                                                       size=(512, 512, 3, 3))
        # self.params['b5_1'] = cp.zeros(512)
        # self.params['beta5_1'] = cp.zeros(2048)  # 存在独立的维度参数  (D,1)  # 2*2*512
        # self.params['gamma5_1'] = cp.ones(2048)


        # full-connected layer
        # self.params['W6_1'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (512 * 1 * 1)) ** 0.5,
        #                                                       size=(256, 512, 3, 3))
        # self.params['b6_1'] = cp.zeros(256)
        # self.params['beta6_1'] = cp.zeros(256)  # 1*1*256
        # self.params['gamma6_1'] = cp.ones(256)

        self.params['W6'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (512 * 4 * 4)) ** 0.5,
                                                            size=(4 * 4 * 512, 10))
        self.params['b6'] = cp.ones(10)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1_1, b1_1 = self.params['W1_1'], self.params['b1_1']
        W1_2, b1_2 = self.params['W1_2'], self.params['b1_2']

        W2_1, b2_1 = self.params['W2_1'], self.params['b2_1']
        W2_2, b2_2 = self.params['W2_2'], self.params['b2_2']

        W3_1, b3_1 = self.params['W3_1'], self.params['b3_1']
        W3_2, b3_2 = self.params['W3_2'], self.params['b3_2']

        W4_1, b4_1 = self.params['W4_1'], self.params['b4_1']
        W4_2, b4_2 = self.params['W4_2'], self.params['b4_2']

        # W5_1, b5_1 = self.params['W5_1'], self.params['b5_1']

        W6, b6 = self.params['W6'], self.params['b6']

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': 1}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # batchnorm
        bn_param1_1 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma1_1'], 'beta': self.params['beta1_1']}
        bn_param1_2 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma1_2'], 'beta': self.params['beta1_2']}
        bn_param2_1 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma2_1'], 'beta': self.params['beta2_1']}
        bn_param2_2 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma2_2'], 'beta': self.params['beta2_2']}
        bn_param3_1 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma3_1'], 'beta': self.params['beta3_1']}
        bn_param3_2 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma3_2'], 'beta': self.params['beta3_2']}
        bn_param4_1 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma4_1'], 'beta': self.params['beta4_1']}
        bn_param4_2 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma4_2'], 'beta': self.params['beta4_2']}
        # bn_param5_1 = {'mode': 'train', 'eps': 1e-5, 'gamma': self.params['gamma5_1'], 'beta': self.params['beta5_1']}


        # compute the forward pass
        feature_1_1, cache_1_1 = conv_bn_relu_forward(X, W1_1, b1_1, conv_param, bn_param1_1)
        feature_1_2, cache_1_2 = conv_bn_relu_forward(feature_1_1, W1_2, b1_2, conv_param, bn_param1_2)
        feature_1_3, cache_1_3 = max_pool_forward(feature_1_2, pool_param)  # 16 * 16 * 64

        feature_2_1, cache_2_1 = conv_bn_relu_forward(feature_1_3, W2_1, b2_1, conv_param, bn_param2_1)  # 128
        feature_2_2, cache_2_2 = conv_bn_relu_forward(feature_2_1, W2_2, b2_2, conv_param, bn_param2_2)
        feature_2_2 = 0.5 * (feature_2_1 + feature_2_2)
        feature_2_3, cache_2_3 = max_pool_forward(feature_2_2, pool_param)


        feature_3_1, cache_3_1 = conv_bn_relu_forward(feature_2_3, W3_1, b3_1, conv_param, bn_param3_1)
        feature_3_2, cache_3_2 = conv_bn_relu_forward(feature_3_1, W3_2, b3_2, conv_param, bn_param3_2)
        feature_3_2 = 0.5 * (feature_3_1 + feature_3_2)
        feature_3_3, cache_3_3 = max_pool_forward(feature_3_2, pool_param)


        feature_4_1, cache_4_1 = conv_bn_relu_forward(feature_3_3, W4_1, b4_1, conv_param, bn_param4_1)
        feature_4_2, cache_4_2 = conv_bn_relu_forward(feature_4_1, W4_2, b4_2, conv_param, bn_param4_2)
        feature_4_2 = 0.5 * (feature_4_1 + feature_4_2)

        if y is not None:
            feature_4_2 = dropout_forward(feature_4_2, 0.65)

        scores, cache_6 = affine_forward(feature_4_2, W6, b6)

        if y is None:
            return scores

        # compute the backward pass
        data_loss, dscores = softmax_loss(scores, y)

        dfeature_6, dW6, db6 = affine_backward(dscores, cache_6)

        # dfeature_5_1, dW5_1, db5_1, dgamma5_1, dbeta5_1 = conv_bn_relu_backward(dfeature_6, cache_5_1)

        # dfeature_4_3 = max_pool_backward(dfeature_5_1, cache_4_3)
        dfeature_4_2, dW4_2, db4_2, dgamma4_2, dbeta4_2 = conv_bn_relu_backward(dfeature_6, cache_4_2)
        dfeature_4_1, dW4_1, db4_1, dgamma4_1, dbeta4_1 = conv_bn_relu_backward(dfeature_4_2, cache_4_1)

        dfeature_3_3 = max_pool_backward(dfeature_4_1, cache_3_3)
        dfeature_3_2, dW3_2, db3_2, dgamma3_2, dbeta3_2 = conv_bn_relu_backward(dfeature_3_3, cache_3_2)
        dfeature_3_1, dW3_1, db3_1, dgamma3_1, dbeta3_1 = conv_bn_relu_backward(dfeature_3_2, cache_3_1)

        dfeature_2_3 = max_pool_backward(dfeature_3_1, cache_2_3)
        dfeature_2_2, dW2_2, db2_2, dgamma2_2, dbeta2_2 = conv_bn_relu_backward(dfeature_2_3, cache_2_2)
        dfeature_2_1, dW2_1, db2_1, dgamma2_1, dbeta2_1 = conv_bn_relu_backward(dfeature_2_2, cache_2_1)

        dfeature_1_3 = max_pool_backward(dfeature_2_1, cache_1_3)
        dfeature_1_2, dW1_2, db1_2, dgamma1_2, dbeta1_2 = conv_bn_relu_backward(dfeature_1_3, cache_1_2)
        dX, dW1_1, db1_1, dgamma1_1, dbeta1_1 = conv_bn_relu_backward(dfeature_1_2, cache_1_1)

        # Add regularization
        N = X.shape[0]

        dW1_1 += self.reg * W1_1
        dW1_2 += self.reg * W1_2

        dW2_1 += self.reg * W2_1
        dW2_2 += self.reg * W2_2

        dW3_1 += self.reg * W3_1
        dW3_2 += self.reg * W3_2

        dW4_1 += self.reg * W4_1
        dW4_2 += self.reg * W4_2


        dW6 += self.reg * W6

        reg_loss = 0.5 * self.reg * sum(cp.sum(W * W) for W in [W1_1, W1_2, W2_1, W2_2, W3_1, W3_2,
                                                                W4_1, W4_2,
                                                                # W5_1,
                                                                # W5_2,
                                                                W6])

        dW1_1 /= N
        dW1_2 /= N
        dW2_1 /= N
        dW2_2 /= N
        dW3_1 /= N
        dW3_2 /= N
        dW4_1 /= N
        dW4_2 /= N
        # dW5_1 /= N
        # dW5_2 /= N

        # dW6_1 /= N
        dW6 /= N

        loss = data_loss + reg_loss

        grads = {'W1_1': dW1_1, 'b1_1': db1_1, 'W1_2': dW1_2, 'b1_2': db1_2,
                 'gamma1_1': dgamma1_1, 'beta1_1': dbeta1_1, 'gamma1_2': dgamma1_2, 'beta1_2': dbeta1_2,
                 'W2_1': dW2_1, 'b2_1': db2_1, 'W2_2': dW2_2, 'b2_2': db2_2,
                 'gamma2_1': dgamma2_1, 'beta2_1': dbeta2_1, 'gamma2_2': dgamma2_2, 'beta2_2': dbeta2_2,
                 'W3_1': dW3_1, 'b3_1': db3_1, 'W3_2': dW3_2, 'b3_2': db3_2,
                 'gamma3_1': dgamma3_1, 'beta3_1': dbeta3_1, 'gamma3_2': dgamma3_2, 'beta3_2': dbeta3_2,
                 'W4_1': dW4_1, 'b4_1': db4_1, 'W4_2': dW4_2, 'b4_2': db4_2,
                 'gamma4_1': dgamma4_1, 'beta4_1': dbeta4_1, 'gamma4_2': dgamma4_2, 'beta4_2': dbeta4_2,
                 # 'W5_1': dW5_1, 'b5_1': db5_1,
                 # 'gamma5_1': dgamma5_1, 'beta5_1': dbeta5_1,

                 'W6': dW6, 'b6': db6
                 }

        return loss, grads
