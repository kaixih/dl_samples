import tensorflow as tf

from tensorflow.python.ops import gen_nn_ops

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

print("Done.")

