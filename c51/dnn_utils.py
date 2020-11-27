import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
import functools


def lazy_property(func):
  attribute = '_lazy_' + func.__name__

  @property
  @functools.wraps(func)
  def wrapper(self):
    if not hasattr(self, attribute):
      setattr(self, attribute, func(self))
    return getattr(self, attribute)
  return wrapper


def conv(inputs, kernel_shape, bias_shape, strides, w_i, b_i=None, activation=tf.nn.relu):
    weights = tf.get_variable('weights', shape=kernel_shape, initializer=w_i)
    conv = tf.nn.conv2d(inputs, weights, strides=strides, padding='SAME')
    if bias_shape is not None:
      biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
      return activation(conv + biases) if activation is not None else conv + biases
    return activation(conv) if activation is not None else conv

  # Use default bias and relu as activation function
def dense(inputs, units, bias_shape, w_i, b_i=None, activation=tf.nn.relu):
    if not isinstance(inputs, ops.Tensor):
      inputs = ops.convert_to_tensor(inputs, dtype='float')
    if len(inputs.shape) > 2:
      inputs = tf.contrib.layers.flatten(inputs) 
    flatten_shape = inputs.shape[1]
    weights = tf.get_variable('weights', shape=[flatten_shape], initializer=w_i)
    dense = tf.matmul(inputs, weights)
    if bias_shape is not None:
      assert bias_shape[0] == units
      biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
      return activation(dense + biases) if activation is not None else dense + biases
    return activation(dense) if activation is not None else dense

