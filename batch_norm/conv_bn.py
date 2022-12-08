import argparse
import numpy as np
import math
import tensorflow as tf
import time
from tensorflow.keras.layers import Conv2D, BatchNormalization

parser = argparse.ArgumentParser(description="Test/Benchmark conv-bn.")
parser.add_argument('-N', type=int, default=2)
parser.add_argument('-H', type=int, default=3)
parser.add_argument('-W', type=int, default=3)
parser.add_argument('-C', type=int, default=8)
parser.add_argument('-K', type=int, default=4)
parser.add_argument('-R', type=int, default=2)
parser.add_argument('-S', type=int, default=2)
parser.add_argument('--bench', default=False, action='store_true')
args, _ = parser.parse_known_args()

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# To make sure we have the same results from C, we use C-like PRNG.
from ctypes import CDLL
libc = CDLL('libc.so.6')
RAND_MAX = 2147483647

# x_shape and k_shape are shapes for input and kernel respectively. Both in NHWC
# formats. g_shape is the shape for gamma/beta/mean/var.
in_features, out_features = args.C, args.K
x_shape = (args.N, args.H, args.W, in_features)
k_shape = (out_features, args.R, args.S, in_features)
g_shape = (out_features, )
conv = Conv2D(filters=out_features, kernel_size=(args.R, args.S),
              padding='valid', use_bias=False)
bn = BatchNormalization()

model = tf.keras.Sequential()
model.add(conv)
model.add(bn)
model.build(x_shape)

# Don't use libc rand for benchmarking since it will be very slow.
def get_random(shape):
  if args.bench:
    return tf.random.normal(shape)
  else:
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

if args.bench:
  warmups = 10
  repeats = 20
  p = tf.constant(0.)

  for i in range(warmups):
    y = model(x)
  p = p + 1.
  p.numpy()

  start = time.time()
  for i in range(repeats):
    y = model(x)
  p = p + 1.
  p.numpy()
  end = time.time()
  time_in_ms = (end - start) / repeats * 1000
  print(f"Results(Fusion): Input: {x_shape} Filter: {k_shape} "
        f"time(ms): {time_in_ms}")
else:
  y = model(x)

if not args.bench:
  print(y)
  print("X shape:", x.shape)
  print("Conv kernel shape:", conv.kernel.shape)
  print("BN gamma shape:", bn.gamma.shape)
  print("BN beta shape:", bn.beta.shape)

