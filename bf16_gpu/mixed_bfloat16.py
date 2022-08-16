import tensorflow as tf

from keras.mixed_precision import autocast_variable
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.python.ops import gen_nn_ops

# This works, meaning the autocast works for fp32->bf16.
#############################
dtype = tf.bfloat16
v = tf.Variable(1.0, dtype=tf.float32)
v = autocast_variable.AutoCastVariable(v)
print("Outside scope: cast_dtype:", v._cast_dtype) # fp32
print("Outside scope: dtype:", v.dtype) # fp32
print("Outside scope: after op dtype:", tf.identity(v).dtype) # fp32
with autocast_variable.enable_auto_cast_variables(dtype):
  print("In scope: cast_dtype:", v._cast_dtype) # fp16/bf16
  print("In scope: dtype:", v.dtype) # fp32
  print("In scope: after op dtype:", tf.identity(v).dtype) # fp16/bf16

# Doesn't work for the mixed_bfloat16 on GPUs. Will get:
#   Incompatible type conversion requested to type 'float32' for
#   AutoCastVariable which is casted to type 'bfloat16'
#############################

if False:
 mixed_precision.set_global_policy('mixed_bfloat16')

 conv2d = layers.Conv2D(filters=4, kernel_size=2)

 x = tf.random.normal(shape=(4, 10, 10, 8))
 y = conv2d(x)
 print(conv2d.dtype_policy)
 print('y.dtype: %s' % y.dtype.name)
 print('conv2d.kernel.dtype: %s' % conv2d.kernel.dtype.name)

 _ = y.numpy()


# When we set the dtype to be bf16, the output y will be an empty tensor. This
# is due to that the Conv2D op is compatible with bfloat16 but no actual
# bfloat16 kernel is registered.
##############################

dtype = tf.bfloat16
x = tf.random.normal(shape=(4, 10, 10, 8), dtype=dtype)
w = tf.random.normal(shape=(2, 2, 8, 4), dtype=dtype)
y = gen_nn_ops.conv2d(x, w, strides=(1, 1, 1, 1), padding='VALID',
                      data_format='NHWC')
_ = y.numpy()
print(y)

# When we set the dtype to be bf16, the matmul works. This is due to the MatMul
# op is compatible with bfloat16 and bfloat16 implementation is in place on CPU.
##############################

dtype = tf.bfloat16
a = tf.random.normal(shape=(4, 16), dtype=dtype)
b = tf.random.normal(shape=(16, 8), dtype=dtype)
c = tf.linalg.matmul(a, b)
_ = c.numpy()
print(c)

print("Done.")
