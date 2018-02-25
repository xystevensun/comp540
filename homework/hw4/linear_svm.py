import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  # print y.shape     (51,)
  # print X.shape     (51, 2)
  # print theta.shape (2,)
  hinge = y*np.matmul(X,theta)
  J = np.sum(np.square(theta))/2/m + np.sum(np.maximum(1-hinge, np.zeros(hinge.shape)))*C/m
  grad = theta/m + np.sum((hinge<1).astype("float")[:,None]*(-y[:,None]*X), axis=0)*C/m
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################
  # print X.shape #(49000, 3073)
  # print y.shape #(49000,)
  # print theta.shape #(3073, 10)
  for mm in range(m):
    p2 = np.dot(theta[:,y[mm]], X[mm,:])
    for yy in range(K):
      if yy != y[mm]:
        m2 = np.dot(theta[:,yy], X[mm,:])-p2+delta
        J += (max(0, m2))
        if m2 > 0:
          dtheta[:,yy] += X[mm,:]
          dtheta[:,y[mm]] -= X[mm,:]
  J = np.sum(np.square(theta))/2/m + J*reg/m
  dtheta = theta/m + dtheta*reg/m
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  # 8-10 lines of code                                                        #
  #############################################################################
  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples
  thetaX = np.matmul(X, theta)
  thetaXthetaX = thetaX - thetaX[range(len(y)),y].reshape((-1,1))
  thetaXthetaX[thetaXthetaX!=0] += delta
  l = np.maximum(0, thetaXthetaX)
  J = np.sum(np.square(theta))/2/m + np.sum(l)*reg/m
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  co = (thetaXthetaX>0).astype(int)
  co[range(len(y)),y] = -(np.sum(co, axis=1))  
  dtheta = theta/m + np.matmul(X.T, co)*reg/m
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta
