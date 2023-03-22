import numpy as np
import cupy as cp


def softmax_loss(X, y):
    # 等同于交叉熵损失
    """
    Computes the loss and gradient for softmax classification.    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = cp.exp(X - cp.max(X, axis=1, keepdims=True))
    probs /= cp.sum(probs, axis=1, keepdims=True)
    N = X.shape[0]
    # 极大似然估计的交叉熵
    loss = -cp.sum(cp.log(probs[cp.arange(N), y])) / N
    
    dx = probs.copy()
    dx[cp.arange(N), y] -= 1

    return loss, dx
  

def cross_entropy_loss(scores, y):
    # softmax
    N, D = scores.shape
    scores -= cp.max(scores, axis=1, keepdims=True)
    fj = cp.exp(scores)
    # 标签类的分数+所有类别分数求和
    data_loss = cp.sum(-scores[cp.arange(N), y] + cp.log(cp.sum(fj, axis=1)))
    # N张图片数据
    data_loss /= N
    # 正则化项
    # data_loss += reg*(cp.sum(W1**2) +cp.sum(W2**2))
    # -cp.log(scores[cp.arange(N), y]/cp.sum(scores, axis=1))  # 每个值的交叉熵损失
    # data_loss = cp.sum(-cp.log(scores[cp.arange(N), y]/cp.sum(scores, axis=1))) / N

    # dscores
    dfj = cp.ones_like(scores)
    dfj = dfj / cp.sum(fj, axis=1, keepdims=True)
    dscores = dfj * cp.exp(scores)
    dscores[cp.arange(N), y] -= 1
    dscores /= N

    return data_loss, dscores