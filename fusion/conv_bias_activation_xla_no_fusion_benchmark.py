import os
os.environ['XLA_FLAGS'] = '--xla_gpu_use_runtime_fusion=false'
os.environ['TF_CUDNN_USE_RUNTIME_FUSION'] = '1'

import argparse
import tensorflow as tf
import time

from tensorflow.keras.layers import Conv2D, ReLU, ELU

# Use mixed precsion to enable the fusion with ELU which requires fp16 inputs.
tf.keras.mixed_precision.set_global_policy('mixed_float16')

parser = argparse.ArgumentParser(description="Test conv-bias-relu fusion.")
parser.add_argument('-x', '--xla', action='store_true',
                    help="""Whether to enable XLA.""")
parser.add_argument('-a', '--activation', type=int, default=0,
                    help="""The activation type.""")
parser.add_argument('-i', '--input', type=str, default="256,64,55,55",
                    help="""The input shape in NCHW.""")
parser.add_argument('-f', '--filter', type=str, default="256,64,3,3",
                    help="""The filter shape in NCHW.""")
args, _ = parser.parse_known_args()
xla_on = args.xla
act_type = args.activation
print(f"Enable XLA: {xla_on}")

if act_type == 0:
  act = ReLU()
elif act_type == 1:
  act = ELU(alpha=1.0)
print(f"Activation Type: {act.name}")

N, C, H, W = (int(n) for n in args.input.split(','))
K, I, R, S = (int(n) for n in args.filter.split(','))
assert(I == C)
print(f"Input shape: {N}, {C}, {H}, {W}")
print(f"Filter shape: {K}, {C}, {R}, {S}")
x_shape = (N, H, W, C)
x = tf.random.normal(x_shape)

conv2d = Conv2D(filters=K, kernel_size=(R, S), padding='same')

@tf.function(jit_compile=xla_on)
def conv_bias_act(x):
  y = conv2d(x)
  y = act(y)
  return y

warmups = 10
repeats = 20
p = tf.constant(0.)

for i in range(warmups):
  y = conv_bias_act(x)
p = p + 1.
p.numpy()

start = time.time()
for i in range(repeats):
  y = conv_bias_act(x)
p = p + 1.
p.numpy()
end = time.time()
time_in_ms = (end - start) / repeats * 1000
print(f"Results(NoFusion): Input: {N},{C},{H},{W} Filter: {K},{C},{R},{S} "
      f"time(ms): {time_in_ms}")

print("Benchmark done!")

