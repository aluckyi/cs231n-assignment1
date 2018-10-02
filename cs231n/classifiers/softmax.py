import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    f_i = X[i].dot(W)
    f_i -= np.max(f_i)
    sum_j = np.sum(np.exp(f_i))
    p = lambda k: np.exp(f_i[k]) / sum_j
    loss += -np.log(p(y[i]))
    for j in range(num_classes):
      dW[:, j] += (p(j) - (j == y[i])) * X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W

  # for i in range(num_train):
  #   # 计算分值向量
  #   f_i = X[i].dot(W)
  #   # 为避免数值不稳定的问题，每个分值向量都减去向量中的最大值
  #   f_i -= np.max(f_i)
  #   # 计算损失值
  #   sum_j = np.sum(np.exp(f_i))
  #   p = lambda k: np.exp(f_i[k]) / sum_j
  #   loss += -np.log(p(y[i]))  # 每一个图像的损失值都要加一起，之后再求均值
  #
  #   # 计算梯度
  #   for k in range(num_classes):
  #     p_k = p(k)
  #     dW[:, k] += (p_k - (k == y[i])) * X[i]
  #
  # loss /= num_train
  # loss += 0.5 * reg * np.sum(W * W)  # 参见知识点中的loss函数公式
  # dW /= num_train
  # dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  f = X.dot(W)
  f -= np.max(f, axis=1, keepdims=True)
  sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
  p = np.exp(f) / sum_f
  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  # gradient
  ind = np.zeros_like(p)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(p - ind)

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

