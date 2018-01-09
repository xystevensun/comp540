from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
import plot_utils


#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    # mu = np.zeros((X.shape[1],))
    # sigma = np.ones((X.shape[1],))
    # X_norm = np.zeros(X.shape)
    mu = np.average(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = np.divide(X-np.tile(mu, (X.shape[0],1)), np.tile(sigma, (X.shape[0],1)))
    ########################################################################
    return X_norm, mu, sigma


#############################################################################
#  Plot the learning curve for training data (X,y) and validation set       #
# (Xval,yval) and regularization lambda reg.                                #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 7 lines of code expected                                                #
    ###########################################################################
    for num_train in range(num_examples):
      reglinear_reg = RegularizedLinearReg_SquaredLoss()
      theta = reglinear_reg.train(X[:num_train+1],y[:num_train+1],reg=reg,num_iters=1000)
      error_train[num_train] = reglinear_reg.loss(theta,X[:num_train+1],y[:num_train+1],0.0)
      error_val[num_train] = reglinear_reg.loss(theta,Xval,yval,0.0)
    ###########################################################################

    return error_train, error_val

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval):
  
  reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
  error_train = np.zeros((len(reg_vec),))
  error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################
  for idx in range(len(reg_vec)):
    reglinear_reg = RegularizedLinearReg_SquaredLoss()
    theta = reglinear_reg.train(X,y,reg=reg_vec[idx],num_iters=1000)
    error_train[idx] = reglinear_reg.loss(theta,X,y,0.0)
    error_val[idx] = reglinear_reg.loss(theta,Xval,yval,0.0)
  return reg_vec, error_train, error_val

import random

#############################################################################
#  Plot the averaged learning curve for training data (X,y) and             #
#  validation set  (Xval,yval) and regularization lambda reg.               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def averaged_learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 10-12 lines of code expected                                            #
    ###########################################################################
    repeat = 50
    for num_train in range(num_examples):
      random_train = range(0, num_examples)
      random_val = range(0, Xval.shape[0])
      for idx in range(repeat):
        random.shuffle(random_train)
        random.shuffle(random_val)
        reglinear_reg = RegularizedLinearReg_SquaredLoss()
        theta = reglinear_reg.train(X[random_train[:num_train+1]],y[random_train[:num_train+1]],reg=reg,num_iters=1000)
        error_train[num_train] += reglinear_reg.loss(theta,X[random_train[:num_train+1]],y[random_train[:num_train+1]],0.0)
        error_val[num_train] += reglinear_reg.loss(theta,Xval[random_val[:max(num_train+1,X.shape[0])]],yval[random_val[:max(num_train+1,X.shape[0])]],0.0)
      error_train[num_train] /= repeat 
      error_val[num_train] /=repeat
    ###########################################################################
    return error_train, error_val


#############################################################################
# Utility functions
#############################################################################
    
def load_mat(fname):
  d = scipy.io.loadmat(fname)
  X = d['X']
  y = d['y']
  Xval = d['Xval']
  yval = d['yval']
  Xtest = d['Xtest']
  ytest = d['ytest']

  # need reshaping!

  X = np.reshape(X,(len(X),))
  y = np.reshape(y,(len(y),))
  Xtest = np.reshape(Xtest,(len(Xtest),))
  ytest = np.reshape(ytest,(len(ytest),))
  Xval = np.reshape(Xval,(len(Xval),))
  yval = np.reshape(yval,(len(yval),))

  return X, y, Xtest, ytest, Xval, yval









