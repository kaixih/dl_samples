import numpy as np
import tensorflow as tf

config = tf.ConfigProto()

with tf.Session(config=config) as sess:

  x = tf.placeholder(tf.float32, [32, 64, 56, 56])
  kernels = tf.Variable(tf.constant(1., shape=[1, 1, 64, 256]))
  hidden = tf.keras.backend.conv2d(x, kernels, data_format='channels_first')

  x_ = np.random.random((32, 64, 56, 56)).astype(np.float32)
  sess.run(tf.global_variables_initializer())

  hidden_ = sess.run([hidden], feed_dict={x: x_})
  print("Test done.")


