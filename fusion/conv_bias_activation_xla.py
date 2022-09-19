import os
os.environ['TF_CUDNN_USE_RUNTIME_FUSION'] = '1'
os.environ['TF_CPP_VMODULE'] = 'cudnn_fused_conv_rewriter=4,remapper=2'

import argparse
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, ReLU, ELU

# Use mixed precsion to enable the fusion with ELU which requires fp16 inputs.
tf.keras.mixed_precision.set_global_policy('mixed_float16')

parser = argparse.ArgumentParser(description="Test conv-bias-relu fusion.")
parser.add_argument('-x', '--xla', action='store_true',
                    help="""Whether to enable XLA.""")
parser.add_argument('-a', '--activation', type=int, default=0,
                    help="""The activation type.""")
args, _ = parser.parse_known_args()
xla_on = args.xla
act_type = args.activation
print(f"Enable XLA: {xla_on}")

if act_type == 0:
  act = ReLU()
elif act_type == 1:
  act = ELU()
print(f"Activation Type: {act.name}")

N, C, H, W = 3, 8, 100, 100
K, C, R, S = 16, C, 3, 3
x_shape = (N, H, W, C)

conv2d = Conv2D(filters=K, kernel_size=(R, S), padding='same')

@tf.function(jit_compile=xla_on)
def conv_bias_act(x):
  y = conv2d(x)
  y = act(y)
  return y

x = tf.random.normal(x_shape)
y = conv_bias_act(x)
_ = y.numpy()
print("Test done!")

