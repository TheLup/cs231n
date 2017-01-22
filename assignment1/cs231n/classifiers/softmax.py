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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    sub_exp_sum = np.sum(np.exp(scores))

    for j in range(num_classes):
        if j == y[i]:
            dW[:, j] += X[i,:].T * (np.exp(correct_class_score) / \
                    sub_exp_sum -1)
        else:
            dW[:, j] += X[i,:].T * (np.exp(scores[j]) / sub_exp_sum)

    loss -= np.log(np.exp(correct_class_score)/sub_exp_sum)

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W


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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)

  num_train_arange = np.arange(num_train)
  correct_class_score = scores[num_train_arange, y]

  score_exp = np.exp(scores)
  score_exp_sum = np.sum(score_exp, axis = 1)
  correct_class_score_exp_sum = np.exp(correct_class_score)

  num_train_arange = np.arange(num_train)
  num_classes_arange = np.arange(num_classes)

  loss_vec = -np.log(correct_class_score_exp_sum / score_exp_sum)
  loss = np.sum(loss_vec)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  mask = np.zeros_like(scores)
  mask =  score_exp / score_exp_sum.reshape(scores.shape[0], 1)
  mask[num_train_arange, y] = correct_class_score_exp_sum / score_exp_sum - 1
  dW += np.matmul(X.T, mask)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

