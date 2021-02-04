import tensorflow as tf

tf.random.set_seed(1)

bn = tf.keras.layers.BatchNormalization(momentum=0.5,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='ones', moving_variance_initializer='ones',
    fused=True)

x0 = tf.random.uniform((2,1,2,3))
print("x NHWC format:", x0.numpy())
print("x NCHW format:", tf.transpose(x0, [0,3,1,2]).numpy())

with tf.GradientTape() as t:
  t.watch(x0)
  y0 = bn(x0, training=True)
  print("y NHWC format:", y0.numpy())
  print("y NCHW format:", tf.transpose(y0, [0,3,1,2]).numpy())
  loss = tf.reduce_sum(y0)
grads = t.gradient(loss, [x0, bn.trainable_variables])
print("dx NHWC format:", grads[0].numpy())
print("dx NCHW format:", tf.transpose(grads[0], [0,3,1,2]).numpy())
print("dscale:", grads[1][0].numpy())
print("doffset:", grads[1][1].numpy())

