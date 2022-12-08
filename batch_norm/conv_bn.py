import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization

# To make sure we have the same results from C, we use C-like PRNG.
from ctypes import CDLL
libc = CDLL('libc.so.6')
RAND_MAX = 2147483647

# x_shape and k_shape are shapes for input and kernel respectively. Both in NHWC
# formats. g_shape is the shape for gamma/beta/mean/var.
in_features, out_features = 8, 4
x_shape = (2, 3, 3, in_features)
k_shape = (out_features, 2, 2, in_features)
g_shape = (out_features, )
conv = Conv2D(filters=out_features, kernel_size=(2, 2), padding='valid',
              use_bias=False)
bn = BatchNormalization()

model = tf.keras.Sequential()
model.add(conv)
model.add(bn)
model.build(x_shape)

def get_random(shape):
  libc.srand(12)
  r = []
  for _ in range(math.prod(shape)):
    r.append(libc.rand() / RAND_MAX)
  return np.array(r).reshape(shape)

x = tf.convert_to_tensor(get_random(x_shape))

conv_kernel = tf.transpose(get_random(k_shape), perm=[1, 2, 3, 0])
bn_gamma = get_random(g_shape)
bn_beta = get_random(g_shape)
bn_moving_mean = np.array([0.] * out_features)
bn_moving_variance = np.array([1.] * out_features)
model.layers[0].set_weights([conv_kernel])
model.layers[1].set_weights([bn_gamma, bn_beta, bn_moving_mean,
                             bn_moving_variance])

y = model(x)
print(y)
print("X shape:", x.shape)
print("Conv kernel shape:", conv.kernel.shape)
print("BN gamma shape:", bn.gamma.shape)
print("BN beta shape:", bn.beta.shape)

