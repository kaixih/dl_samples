import tensorflow as tf

tf.random.set_seed(1)

x = tf.random.uniform((2,1,2,3))
scale = tf.constant([1.0,1.0,1.0])
offset = tf.constant([0.0,0.0,0.0])
mean = tf.constant([1.0,1.0,1.0])
variance = tf.constant([1.0,1.0,1.0])

y, moving_mean, moving_var, r1, r2, r3 = tf.raw_ops.FusedBatchNormV3(
    x=x, scale=scale, offset=offset, mean=mean, variance=variance,
    epsilon=0.001, exponential_avg_factor=0.5, data_format='NHWC',
    is_training=True, name=None)
print("y:", y.numpy())
print("moving_mean:", moving_mean.numpy())
print("moving_var:", moving_var.numpy())
print("saved mean:", r1.numpy())
print("saved inv var:", r2.numpy())

dy = tf.ones((2,1,2,3))
dx, dscale, doffset, r4, r5 = tf.raw_ops.FusedBatchNormGradV3(
    y_backprop=dy, x=x, scale=scale, reserve_space_1=r1, reserve_space_2=r2,
    reserve_space_3=r3, epsilon=0.001, data_format='NHWC', is_training=True,
    name=None)
print("dx:", dx.numpy())
print("dscale:", dscale.numpy())
print("doffset:", doffset.numpy())
