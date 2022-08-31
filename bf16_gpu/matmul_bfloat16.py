import tensorflow as tf

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
