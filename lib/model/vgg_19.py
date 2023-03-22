from lib.utils.modules import *
from lib.utils.layers import *
from lib.utils.loss import *



class ThreeLayerConvNet(object):
    """    
    A three-layer convolutional network with the following architecture:       
       conv - relu - 2x2 max pool - affine - relu - affine - softmax
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1, reg=0.0,
                 dtype=cp.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize weights and biases
        C, H, W = input_dim
        # 32 * 32
        self.params['W1_1'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (3 * 32 * 32)) ** 0.5, size=(64, 3, 3, 3))
        self.params['W1_2'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (64 * 32 * 32)) ** 0.5, size=(64, 64, 3, 3))
        self.params['b1_1'] = cp.zeros(64)
        self.params['b1_2'] = cp.zeros(64)

        # 16 * 16
        self.params['W2_1'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (64 * 16 * 16)) ** 0.5, size=(128, 64, 3, 3))
        self.params['W2_2'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (128 * 16 * 16)) ** 0.5, size=(128, 128, 3, 3))
        self.params['b2_1'] = cp.zeros(128)
        self.params['b2_2'] = cp.zeros(128)

        # 8 * 8
        self.params['W3_1'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (128 * 8 * 8)) ** 0.5, size=(256, 128, 3, 3))
        self.params['W3_2'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (256 * 8 * 8)) ** 0.5, size=(256, 256, 3, 3))
        self.params['W3_3'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (256 * 8 * 8)) ** 0.5, size=(256, 256, 3, 3))
        self.params['b3_1'] = cp.zeros(256)
        self.params['b3_2'] = cp.zeros(256)
        self.params['b3_3'] = cp.zeros(256)
        
        # 4 * 4
        self.params['W4_1'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (256 * 4 * 4)) ** 0.5, size=(512, 256, 3, 3))
        self.params['W4_2'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (512 * 4 * 4)) ** 0.5, size=(512, 512, 3, 3))
        self.params['W4_3'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (512 * 4 * 4)) ** 0.5, size=(512, 512, 3, 3))
        self.params['b4_1'] = cp.zeros(512)
        self.params['b4_2'] = cp.zeros(512)
        self.params['b4_3'] = cp.zeros(512)
        
        # 2 * 2
        self.params['W5_1'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (512 * 2 * 2)) ** 0.5, size=(512, 512, 3, 3))
        self.params['W5_2'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (512 * 2 * 2)) ** 0.5, size=(512, 512, 3, 3))
        self.params['W5_3'] = weight_scale * cp.random.normal(loc=0.0, scale=(2 / (512 * 2 * 2)) ** 0.5, size=(512, 512, 3, 3))
        self.params['b5_1'] = cp.zeros(512)
        self.params['b5_2'] = cp.zeros(512)
        self.params['b5_3'] = cp.zeros(512)
        
        
        # full-connected layer
        self.params['W6'] = weight_scale * cp.random.normal(loc=0, scale=(2 / (512 * 1 * 1)) ** 0.5, size=(1 * 1 * 512, 10))
        self.params['b6'] = cp.zeros(10)


        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1_1, b1_1 = self.params['W1_1'], self.params['b1_1']
        W1_2, b1_2 = self.params['W1_2'], self.params['b1_2']
        
        W2_1, b2_1 = self.params['W2_1'], self.params['b2_1']
        W2_2, b2_2 = self.params['W2_2'], self.params['b2_2']
        
        W3_1, b3_1 = self.params['W3_1'], self.params['b3_1']
        W3_2, b3_2 = self.params['W3_2'], self.params['b3_2']
        W3_3, b3_3 = self.params['W3_3'], self.params['b3_3']
        
        W4_1, b4_1 = self.params['W4_1'], self.params['b4_1']
        W4_2, b4_2 = self.params['W4_2'], self.params['b4_2']
        W4_3, b4_3 = self.params['W4_3'], self.params['b4_3']

        W5_1, b5_1 = self.params['W5_1'], self.params['b5_1']
        W5_2, b5_2 = self.params['W5_2'], self.params['b5_2']
        W5_3, b5_3 = self.params['W5_3'], self.params['b5_3']
        
        W6, b6 = self.params['W6'], self.params['b6']

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': 1}


        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        # compute the forward pass
        feature_1_1, cache_1_1 = conv_relu_forward(X, W1_1, b1_1, conv_param)
        feature_1_2, cache_1_2 = conv_relu_forward(feature_1_1, W1_2, b1_2, conv_param)
        feature_1_3, cache_1_3 = max_pool_forward (feature_1_2, pool_param)
        
        feature_2_1, cache_2_1 = conv_relu_forward(feature_1_3, W2_1, b2_1, conv_param)
        feature_2_2, cache_2_2 = conv_relu_forward(feature_2_1, W2_2, b2_2, conv_param)
        feature_2_3, cache_2_3 = max_pool_forward (feature_2_2, pool_param)
        
        feature_3_1, cache_3_1 = conv_relu_forward(feature_2_3, W3_1, b3_1, conv_param)
        feature_3_2, cache_3_2 = conv_relu_forward(feature_3_1, W3_2, b3_2, conv_param)
        feature_3_3, cache_3_3 = conv_relu_forward(feature_3_2, W3_3, b3_3, conv_param)
        feature_3_4, cache_3_4 = max_pool_forward (feature_3_3, pool_param)
        
        feature_4_1, cache_4_1 = conv_relu_forward(feature_3_4, W4_1, b4_1, conv_param)
        feature_4_2, cache_4_2 = conv_relu_forward(feature_4_1, W4_2, b4_2, conv_param)
        feature_4_3, cache_4_3 = conv_relu_forward(feature_4_2, W4_3, b4_3, conv_param)
        feature_4_4, cache_4_4 = max_pool_forward (feature_4_3, pool_param)

        feature_5_1, cache_5_1 = conv_relu_forward(feature_4_4, W5_1, b5_1, conv_param)
        feature_5_2, cache_5_2 = conv_relu_forward(feature_5_1, W5_2, b5_2, conv_param)
        feature_5_3, cache_5_3 = conv_relu_forward(feature_5_2, W5_3, b5_3, conv_param)
        feature_5_4, cache_5_4 = max_pool_forward (feature_5_3, pool_param)

        if y is not None:
            #  Dropout
            feature_5_4 = dropout_forward(feature_5_4, 0.6)

        scores, cache_6 = affine_forward(feature_5_4, W6, b6)

        if y is None:
            return scores

        # compute the backward pass
        data_loss, dscores = softmax_loss(scores, y)

        dfeature_6, dW6, db6 = affine_backward(dscores, cache_6)
        
        dfeature_5_4 = max_pool_backward(dfeature_6, cache_5_4)
        dfeature_5_3, dW5_3, db5_3 = conv_relu_backward(dfeature_5_4, cache_5_3)
        dfeature_5_2, dW5_2, db5_2 = conv_relu_backward(dfeature_5_3, cache_5_2)
        dfeature_5_1, dW5_1, db5_1 = conv_relu_backward(dfeature_5_2, cache_5_1)
        
        
        dfeature_4_4 = max_pool_backward(dfeature_5_1, cache_4_4)
        dfeature_4_3, dW4_3, db4_3 = conv_relu_backward(dfeature_4_4, cache_4_3)
        dfeature_4_2, dW4_2, db4_2 = conv_relu_backward(dfeature_4_3, cache_4_2)
        dfeature_4_1, dW4_1, db4_1 = conv_relu_backward(dfeature_4_2, cache_4_1)
    
        dfeature_3_4 = max_pool_backward(dfeature_4_1, cache_3_4)
        dfeature_3_3, dW3_3, db3_3 = conv_relu_backward(dfeature_3_4, cache_3_3)
        dfeature_3_2, dW3_2, db3_2 = conv_relu_backward(dfeature_3_3, cache_3_2)
        dfeature_3_1, dW3_1, db3_1 = conv_relu_backward(dfeature_3_2, cache_3_1)
        
        dfeature_2_3 = max_pool_backward(dfeature_3_1, cache_2_3)
        dfeature_2_2, dW2_2, db2_2 = conv_relu_backward(dfeature_2_3, cache_2_2)
        dfeature_2_1, dW2_1, db2_1 = conv_relu_backward(dfeature_2_2, cache_2_1)
        
        dfeature_1_3 = max_pool_backward(dfeature_2_1, cache_1_3)
        dfeature_1_2, dW1_2, db1_2 = conv_relu_backward(dfeature_1_3, cache_1_2)
        dX, dW1_1, db1_1 = conv_relu_backward(dfeature_1_2, cache_1_1)


        # Add regularization
        N = X.shape[0]
        
        dW1_1 += self.reg * W1_1
        dW1_2 += self.reg * W1_2
        
        dW2_1 += self.reg * W2_1
        dW2_2 += self.reg * W2_2

        dW3_1 += self.reg * W3_1
        dW3_2 += self.reg * W3_2
        dW3_3 += self.reg * W3_3
        
        dW4_1 += self.reg * W4_1
        dW4_2 += self.reg * W4_2
        dW4_3 += self.reg * W4_3
        
        dW5_1 += self.reg * W5_1
        dW5_2 += self.reg * W5_2
        dW5_3 += self.reg * W5_3
        
        dW6 += self.reg * W6
        
        # 引入L2正则化
        reg_loss = 0.5 * self.reg * sum(cp.sum(W * W) for W in [W1_1, W1_2, W2_1, W2_2, W3_1, W3_2, W3_3,
                                                                W4_1, W4_2, W4_3, W5_1, W5_2, W5_3, W6])

        dW1_1 /= N
        dW1_2 /= N
        dW2_1 /= N
        dW2_2 /= N
        dW3_1 /= N
        dW3_2 /= N
        dW3_3 /= N
        dW4_1 /= N
        dW4_2 /= N
        dW4_3 /= N
        dW5_1 /= N
        dW5_2 /= N
        dW5_3 /= N         
  
        dW6 /= N

        loss = data_loss + reg_loss

        grads = {'W1_1': dW1_1, 'b1_1': db1_1, 'W1_2': dW1_2, 'b1_2': db1_2, 
                 'W2_1': dW2_1, 'b2_1': db2_1, 'W2_2': dW2_2, 'b2_2': db2_2,
                 'W3_1': dW3_1, 'b3_1': db3_1, 'W3_2': dW3_2, 'b3_2': db3_2, 'W3_3': dW3_3, 'b3_3': db3_3, 
                 'W4_1': dW4_1, 'b4_1': db4_1, 'W4_2': dW4_2, 'b4_2': db4_2, 'W4_3': dW4_3, 'b4_3': db4_3, 
                 'W5_1': dW5_1, 'b5_1': db5_1, 'W5_2': dW5_2, 'b5_2': db5_2, 'W5_3': dW5_3, 'b5_3': db5_3, 
                 'W6': dW6, 'b6': db6}
                 

        return loss, grads