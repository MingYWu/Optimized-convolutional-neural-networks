import pickle
from lib.utils import optim
import cupy as cp
import numpy as np
from imgaug import augmenters as iaa  #引入数据增强的包


class Solver(object):

    def __init__(self, model, data, num_epochs, batch_size, lr_decay, lr_decay_epochs, **kwargs):
        self.model = model
        self.best_acc_model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        # 验证集
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        # 获取基本的超参数
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.lr = 0

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)   # 设置运行的时候显示详细信息
        
        # 数据增强
        self.imgaug = iaa.Sequential([         # 建立一个名为seq的实例，定义增强方法，用于增强
                iaa.Crop(px=(0, 16)),          # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
                iaa.Fliplr(0.5),               # 对百分之五十的图像进行做左右翻转 对50%的图像进行镜像翻转
                iaa.Flipud(0.2),               # 左右翻转
                iaa.GaussianBlur((0, 1.0)),    # 在模型上使用0均值1方差进行高斯模糊
                # iaa.AdditiveGaussianNoise(     # 加入高斯噪声
                #     loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                # ),
            ])

        # Throw an error if there are extra keyword arguments
        # 不能识别的参数信息报错
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        # 当不存在优化策略时
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        
        # getattr() 函数用于返回一个对象属性值
        self.update_rule = getattr(optim, self.update_rule)  
        self._reset()
        

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []   # 损失函数集合
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        选择数据，
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        # 从num_train 中随机抽取数字，返回指定大小(size)的数组 batch_size
        batch_mask = np.random.choice(num_train, self.batch_size)

        # 建立一个和x_train一样的 X_batch，深度复制
        X_batch = self.X_train[batch_mask].copy()
        y_batch = self.y_train[batch_mask].copy()
        # print(X_batch.shape) # (128, 32, 32, 3)
        
        # 数据增强
        # for i in range(X_batch.shape[0]):
            # X_batch[i] = self.imgaug(image=X_batch[i])['image']
            # X_batch[i] = self.imgaug.augment_batch(X_batch[i], background=True)
        # X_batch = self.imgaug.augment_images(images=X_batch)

        # 转化成GPU能运算的格式，图像进行reshape，对于图像，转化为能放入卷积神经网络中训练的格式
        # print(X_batch[0].shape)
        X_batch = cp.asarray(X_batch).transpose(0, 3, 1, 2)
        y_batch = cp.asarray(y_batch)

        # Compute loss and gradient 计算梯度和损失
        # 计算损失，返回loss值和权重，偏置的字典
        loss, grads = self.model.loss(X_batch, y_batch)
        # 存储计算出来新的loss
        self.loss_history.append(loss)

        # Perform a parameter update，更新参数
        for p, w in self.model.params.items():
            # p=key，w=参数值
            dw = grads[p]
            config = self.optim_configs[p]
            # adam, sgd, 选择不同的梯度优化算法
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=2):
        # 检查准确率
        # Maybe subsample the data
        N = X.shape[0]                                   # 训练集上总样本数量 450000
        acc = 0
        if num_samples is not None and N > num_samples:  # 样本总量大于采样的数量
            mask = np.random.choice(N, num_samples)      # 随机采取4张照片计算  在[0, N)内输出五个数字并组成一维数组（ndarray）
            N = num_samples
            X = X[mask]
            y = y[mask]
    
        X = cp.asarray(X).transpose(0, 3, 1, 2)  # 转化用于后面的loss计算
        y = cp.asarray(y)   # 样本的真实标签类别
        # print(X.shape)
    
        # Compute predictions in batches
        num_batches = int(N / batch_size)  # 100 / 2
        if N % batch_size != 0:
            num_batches += 1
        
        y_pred = []
        # 计算每一个batch下的准确度
        for i in range(num_batches):
            start = i * batch_size                    # 图像索引起
            end = (i + 1) * batch_size                # 图像索引止，小批量下的计算
            scores = self.model.loss(X[start:end])    # shape = (batch_size, 10)
            y_pred.append(np.argmax(scores, axis=1))  # 获得scores中最大的值，同时获得该值的下标索引 == 预测的类别标签
        
        # np.hstack将参数元组的元素数组按水平方向进行叠加，用于和y真实值进行比对
        y_pred = cp.hstack(y_pred)   
        acc = cp.mean(y_pred == y)

        return acc


    def train(self):
        """
        Run optimization to train the model_cifar10.
        """
        num_train = self.X_train.shape[0]                            # 训练集的总张数：45000
        iterations_per_epoch = max(num_train // self.batch_size, 1)  # 迭代次数 ： 45000// 128 = 351
        num_iterations = self.num_epochs * iterations_per_epoch      # 总共的迭代次数：70 * 351
        
        init_lr = self.optim_configs['W1_1']['learning_rate']        # 初始学习率，为了后期进行指数的学习率下降

        for t in range(num_iterations):
            # 迭代次数
            # 更新所有可学习的参数，权重等
            self._step()
        
            # Maybe print training loss
            # 损失函数变化展示
            if self.verbose and t % self.print_every == 0:
                print('[Epoch %d] Iteration: %d / %d, training loss: %f'
                      % (self.epoch + 1, (t + 1) % iterations_per_epoch, iterations_per_epoch, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay the learning rate.
            # 每一轮结束：bool
            epoch_end = (t + 1) % iterations_per_epoch == 0  
            if epoch_end:
                self.epoch += 1
                
                # if self.epoch < 20:
                #     for k in self.optim_configs:
                #         self.optim_configs[k]['learning_rate'] *= self.lr_decay
                #         self.lr = self.optim_configs[k]['learning_rate'] 
                # elif (self.epoch % 20 == 0) and 20 <= self.epoch:
                #      for k in self.optim_configs:
                #         self.optim_configs[k]['learning_rate'] *= 0.5 
                #         self.lr = self.optim_configs[k]['learning_rate'] 
                     
                # 指数衰减的学习率更新方法
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] = init_lr * (self.lr_decay **(self.epoch/self.lr_decay_epochs))
                    self.lr = self.optim_configs[k]['learning_rate']
                
                # 手动调整
                # # 调整学习率
                # if self.epoch % 5 == 0 and self.epoch < 30:
                #     for k in self.optim_configs:
                #         self.optim_configs[k]['learning_rate'] *= 0.8
                #         lr = self.optim_configs[k]['learning_rate'] 
                #         print("学习率调整为：%f" %(lr))
                # elif self.epoch % (self.lr_decay_epochs) == 0 and 30 <= self.epoch <= 60:
                #     for k in self.optim_configs:
                #         self.optim_configs[k]['learning_rate'] *= 0.6
                #         lr = self.optim_configs[k]['learning_rate'] 
                #         print("学习率调整为：%f" %(lr))
                # elif self.epoch % (self.lr_decay_epochs) == 0 and 60 < self.epoch:
                #     for k in self.optim_configs:
                #         self.optim_configs[k]['learning_rate'] *= 0.4
                #         lr = self.optim_configs[k]['learning_rate'] 
                #         print("学习率调整为：%f" %(lr))
                # elif self.epoch % (self.lr_decay_epochs) == 0 and 60 < self.epoch:
                #     for k in self.optim_configs:
                #         self.optim_configs[k]['learning_rate'] *= 0.4
                #         lr = self.optim_configs[k]['learning_rate'] 
                #         print("学习率调整为：%f" %(lr))
                    

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            last_iter = (t == num_iterations - 1)  # 最后一轮
            
            # 第一轮（初始权重下的数据）或者最后一轮或者每一轮的结束
            if last_iter or epoch_end:
                # 训练集准确率
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=5000,
                                                batch_size=self.batch_size)
                # num_samples 用于做acc计算的样本数量
                # print(train_acc)
                # (训练集的测试集)验证集准确率
                val_acc = self.check_accuracy(self.X_val, self.y_val, num_samples=5000,
                                              batch_size=self.batch_size)
                # print(val_acc)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                # Keep track of the best model_cifar10
                # 为了去保存最好的模型权重，其中保存在验证集上准确度已经上80的权重
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

                    # 保存最好的参数
                    import datetime
                    import os

                    self.best_acc_model.params = self.best_params
                    print(f'current_best_acc: {val_acc}')
                    model_file = './save_pth'
                    if not os.path.exists(model_file):
                        os.mkdir(model_file)
                    # overwrite the params
                    val_acc_2 = str(val_acc)
                    # model_path = model_file + f'/model_{datetime.date.today()}_{val_acc_2}.pkl'
                    if val_acc > 0.80:
                        model_path = model_file + f'/model_{datetime.date.today()}_{val_acc_2}.pkl'
                        model_params_pth = open(model_path, 'wb')
                        pickle.dump(self.best_acc_model, model_params_pth)
                        model_params_pth.close()
                        print(f"{datetime.date.today()}: save the model_cifar10 param to : {model_path}")

                # if self.verbose and epoch_end:
                if self.verbose and epoch_end:
                    # 读取训练以及测试误差
                    print('[Epoch %d/%d] train acc: %.6f; val_acc: %.6f, best_val_acc: %f'
                          % (self.epoch, self.num_epochs, train_acc, val_acc, self.best_val_acc))
                    print("学习率调整为：%f" %(self.lr))

        self.model.params = self.best_params
