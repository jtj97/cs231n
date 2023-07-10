from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    # forward
    for i in range(num_train):
      # 对train样本进行遍历
      scores = X[i].dot(W) # 分子 (10,), W(num_train, 10), dSyi
      exp_scores = np.exp(scores)
      exp_scores_sum = np.sum(exp_scores) # 分母
      
      true_prob = exp_scores[y[i]] / exp_scores_sum # Pyi
      loss -= np.log(true_prob) # 真值的prob取-log # Li
      
      # backward
      for j in range(num_classes):
        dLi_dPyi = -1 / true_prob
        # dPyi_dSyj求导有2种情况
        if j==y[i]:
          dPyi_dSyj = ((exp_scores[y[i]] * exp_scores_sum) - exp_scores[y[i]] ** 2) / (exp_scores_sum ** 2)
        else:
          dPyi_dSyj = ( - exp_scores[y[i]] * exp_scores[j]) / (exp_scores_sum ** 2)
        dSyi_dW = X[i]
        dL_dW = dLi_dPyi * dPyi_dSyj * dSyi_dW
        dW[:, j] += dL_dW
    
      
        
    # 均值和正则化
    loss /= num_train
    loss += reg * np.sum(W * W)
        
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    
    # forward
    
    scores = X.dot(W) # (num_train, num_classes) dSy
    exp_scores = np.exp(scores) # (num_train, num_classes)
    exp_scores_sum = np.sum(exp_scores, axis=1) # 分母 (num_train,)
    true_prob = exp_scores[np.arange(num_train), y] / exp_scores_sum # Py (num_train,)
    loss -= np.sum(np.log(true_prob)) # 真值的prob取-log # L

    # backward
    
    dL_dSj = (exp_scores.T / exp_scores_sum).T  # 根据L_i = -f_{y_i} + log \sum_{j}e^{f_j} 先算出L_i右半部分的导数
    dL_dSj[np.arange(num_train), y] -= 1 # 求导有2种情况，如果对y_i求导要多减去1，即L_i左半部分的导数
    dW += X.T.dot(dL_dSj) # 链式法则
    
    # 均值和正则化
    loss /= num_train
    loss += reg * np.sum(W * W)
        
    dW /= num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
