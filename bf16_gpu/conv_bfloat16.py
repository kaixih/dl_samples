import argparse
import sys
import tensorflow as tf

from tensorflow.python.ops import gen_nn_ops

parser = argparse.ArgumentParser(description="Test BF16 convolution.")
parser.add_argument('-k', '--kind', type=int,
                    help="""The convolution direction.""")
parser.add_argument('-d', '--dimension', type=int,
                    help="""The convolution dimension.""")

if len(sys.argv) == 1:
  parser.print_help(sys.stderr)
  sys.exit(1)

args, _ = parser.parse_known_args()

kind = args.kind if args.kind else 0
ndims = args.dimension if args.dimension else 2

# Generally, the ConvXX ops have already registered bf16 as a valid dtype.
# However, no GPU kernel is registered.
# Conv2D => empty tensor (no bf16 kernel is registered)
# Conv3D => bf16 tensor (has bf16 kernel only for CPUs)
##############################

N, D, H, W, C = 3, 10, 10, 10, 4
K, T, R, S = 4, 2, 2, 2
dtype = tf.bfloat16

if ndims == 2:
  x_shape = (N, H, W, C)
  f_shape = (R, S, C, K)
  strides = (1, 1, 1, 1)
  data_format = 'NHWC'
  print_shape = (0, 0, 0, slice(None))
else:
  x_shape = (N, D, H, W, C)
  f_shape = (T, R, S, C, K)
  strides = (1, 1, 1, 1, 1)
  data_format = 'NDHWC'
  print_shape = (0, 0, 0, 0, slice(None))

if kind == 0:
  x = tf.random.normal(shape=x_shape, dtype=dtype)
  w = tf.random.normal(shape=f_shape, dtype=dtype)
  conv_fn = gen_nn_ops.conv2d if ndims == 2 else gen_nn_ops.conv3d
  y = conv_fn(x, w, strides=strides, padding='SAME', data_format=data_format)
  y_np = y.numpy()
  print(y_np[print_shape])
elif kind == 1:
  x = tf.random.normal(shape=x_shape, dtype=dtype)
  dy = tf.random.normal(shape=x_shape, dtype=dtype)
  conv_fn = (gen_nn_ops.conv2d_backprop_filter if ndims == 2 else
             gen_nn_ops.conv3d_backprop_filter_v2)
  dw = conv_fn(x, f_shape, dy, strides=strides, padding='SAME',
               data_format=data_format)
  dw_np = dw.numpy()
  print(dw_np[print_shape])
elif kind == 2:
  w = tf.random.normal(shape=f_shape, dtype=dtype)
  dy = tf.random.normal(shape=x_shape, dtype=dtype)
  conv_fn = (gen_nn_ops.conv2d_backprop_input if ndims == 2 else
             gen_nn_ops.conv3d_backprop_input_v2)
  dx = conv_fn(x_shape, w, dy, strides=strides, padding='SAME',
               data_format=data_format)
  dx_np = dx.numpy()
  print(dx_np[print_shape])

print("Done.")

