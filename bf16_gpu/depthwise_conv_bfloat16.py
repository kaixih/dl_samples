import argparse
import sys
import tensorflow as tf

from tensorflow.python.ops import gen_nn_ops

parser = argparse.ArgumentParser(description="Test BF16 depthwise convolution.")
parser.add_argument('-k', '--kind', type=int, default=0,
                    help="The depthwise convolution direction. "
                         "(0=fwd, 1=bwd_filter, 2=bwd_input)")
parser.add_argument('-p', '--path', type=int, default=0,
                    help="Test the execution path. (0=cuda, 1=cudnn_grouped, "
                         "2=cudnn, 3=cublaslt)")

if len(sys.argv) == 1:
  parser.print_help(sys.stderr)
  sys.exit(1)

args, _ = parser.parse_known_args()

N, H, W = 3, 10, 10
K = 1
dtype = tf.bfloat16

if args.path == 0:
  # Execution path for native CUDA kernels.
  x_shape = (N, H, W, 4)
  f_shape = (2, 2, 4, K)
elif args.path == 1:
  # Execution path for cuDNN grouped conv.
  x_shape = (N, H, W, 4)
  f_shape = (3, 3, 4, K)
elif args.path == 2:
  # Execution path for regular Conv2D -> cuDNN.
  x_shape = (N, H, W, 1)
  f_shape = (3, 3, 1, K)
elif args.path == 3:
  # Execution path for regular Conv2D -> cuBlaslt.
  x_shape = (N, H, W, 1)
  f_shape = (1, 1, 1, K)

strides = (1, 1, 1, 1)
data_format = 'NHWC'
print_shape = (0, 0, 0, slice(None))

# depthwise fwd => bf16 tensor (has bf16 kernel only for CPUs)
# depthwise bwd filter => bf16 tensor (has bf16 kernel only for CPUs)
# depthwise bwd input => bf16 tensor (has bf16 kernel only for CPUs)
##############################

if args.kind == 0:
  x = tf.random.normal(shape=x_shape, dtype=dtype)
  w = tf.random.normal(shape=f_shape, dtype=dtype)
  conv_fn = gen_nn_ops.depthwise_conv2d_native
  y = conv_fn(x, w, strides=strides, padding='SAME', data_format=data_format)
  y_np = y.numpy()
  print(y_np[print_shape])
elif args.kind == 1:
  x = tf.random.normal(shape=x_shape, dtype=dtype)
  dy = tf.random.normal(shape=x_shape, dtype=dtype)
  conv_fn = gen_nn_ops.depthwise_conv2d_native_backprop_filter
  dw = conv_fn(x, f_shape, dy, strides=strides, padding='SAME',
               data_format=data_format)
  dw_np = dw.numpy()
  print(dw_np[print_shape])
elif args.kind == 2:
  w = tf.random.normal(shape=f_shape, dtype=dtype)
  dy = tf.random.normal(shape=x_shape, dtype=dtype)
  conv_fn = gen_nn_ops.depthwise_conv2d_native_backprop_input
  dx = conv_fn(x_shape, w, dy, strides=strides, padding='SAME',
               data_format=data_format)
  dx_np = dx.numpy()
  print(dx_np[print_shape])

print("Done.")

