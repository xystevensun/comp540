import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys      #
    # 'theta1' and 'theta1_0'; use keys 'theta2' and 'theta2_0' for the        #
    # weights and biases of the hidden affine layer, and keys 'theta3' and     #
    # 'theta3_0' for the weights and biases of the output affine layer.        #
    ############################################################################
    # about 12 lines of code
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax
    conv_stride = 1
    conv_pad = (filter_size - 1) / 2
    pool_height = 2
    pool_width = 2
    pool_stride = 2
    conv_H = 1 + (input_dim[1] + 2 * conv_pad - filter_size) / conv_stride
    conv_W = 1 + (input_dim[2] + 2 * conv_pad - filter_size) / conv_stride
    HH = 1 + (conv_H - pool_height)/pool_stride
    WW = 1 + (conv_W - pool_width)/pool_stride
    fc_dim = HH*WW*num_filters
    self.params['theta1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size)).astype(dtype)
    self.params['theta1_0'] = np.zeros((num_filters, ), dtype=dtype)
    self.params['theta2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(fc_dim, hidden_dim)).astype(dtype)
    self.params['theta2_0'] = np.zeros((hidden_dim, ), dtype=dtype)
    self.params['theta3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes)).astype(dtype)
    self.params['theta3_0'] = np.zeros((num_classes, ), dtype=dtype)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
    theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
    theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = theta1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # about 3 lines of code (use the helper functions in layer_utils.py)
    
    # def conv_relu_pool_forward(x, theta, theta0, conv_param, pool_param): -> return out, cache
    # def affine_relu_forward(x, theta, theta0): -> return out, cache
    # def affine_forward(x, theta, theta0): -> return out, cache
    out1, cache1 = conv_relu_pool_forward(X, theta1, theta1_0, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, theta2, theta2_0)
    scores, cache3 = affine_forward(out2, theta3, theta3_0)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # about 12 lines of code
    loss, dx = softmax_loss(scores, y)
    reg_term = sum([np.sum(v**2) for k, v in self.params.iteritems()])
    loss += self.reg * reg_term/2

    dx, dtheta, dtheta0 = affine_backward(dx, cache3)
    grads['theta3'] = dtheta + self.reg * self.params['theta3']
    grads['theta3_0'] = dtheta0

    dx, dtheta, dtheta0 = affine_relu_backward(dx, cache2)
    grads['theta2'] = dtheta + self.reg * self.params['theta2']
    grads['theta2_0'] = dtheta0

    dx, dtheta, dtheta0 = conv_relu_pool_backward(dx, cache1)
    grads['theta1'] = dtheta + self.reg * self.params['theta1']
    grads['theta1_0'] = dtheta0
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

