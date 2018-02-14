import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  dp = np.dot(X,theta)
  dp = np.exp(dp-np.max(dp, axis=1).reshape((-1,1)))
  logdp = dp/np.sum(dp, axis=1).reshape((-1,1))
  for m_ in range(m):
    for k_ in range(theta.shape[1]):
      if y[m_] == k_:
        J += np.log(logdp[m_,k_])
  J = -J/m
  for d_ in range(dim):
    for k_ in range(theta.shape[1]):
      J += reg/2/m*theta[d_,k_]*theta[d_,k_]

  for k_ in range(theta.shape[1]):
    for m_ in range(m):
      grad[:, k_] += X[m_,:]*((1 if y[m_]==k_ else 0)-logdp[m_,k_])
    grad[:,k_] /= (-m)
    grad[:,k_] += reg/m*theta[:,k_]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  dp = np.dot(X,theta)
  dp = np.exp(dp-np.max(dp, axis=1).reshape((-1,1)))
  logdp = dp/np.sum(dp, axis=1).reshape((-1,1))
  indicator = np.zeros((m,theta.shape[1]))
  indicator[range(0,m), y] = 1
  J = -np.sum(np.sum(np.multiply(indicator, np.log(logdp))))/m + reg/2.0/m*np.sum(np.sum(np.square(theta)))
  grad = -np.matmul(X.T, (indicator-logdp))/m + reg/m*theta    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
