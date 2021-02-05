import tensorflow as tf

tf.random.set_seed(1)

# When momentum=0.0, the moving_mean/var equals to current batch_mean/var.
# When fused=True, there is no guarantee of bitwise equality.
bn = tf.keras.layers.BatchNormalization(momentum=0.5,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='ones', moving_variance_initializer='ones',
    fused=False)

# Training Step 0 (skip the real scale/offset update)
x0 = tf.random.uniform((2,1,2,3))
print("x(step0):", x0.numpy())

y0 = bn(x0, training=True)
print("moving_mean(step0): ", bn.moving_mean.numpy())
print("moving_var(step0): ", bn.moving_variance.numpy())
print("y(step0):", y0.numpy())

# Training Step 1 (skip the real scale/offset update)
x1 = tf.random.uniform((2,1,2,3))
print("x(step1):", x1.numpy())

y1 = bn(x1, training=True)
print("moving_mean(step1): ", bn.moving_mean.numpy())
print("moving_var(step1): ", bn.moving_variance.numpy())
print("y(step1):", y1.numpy())

# Inference Step
x_infer = tf.random.uniform((2,1,2,3))
print("x(infer):", x_infer.numpy())

y_infer = bn(x_infer, training=False)
print("estimated_mean(infer): ", bn.moving_mean.numpy())
print("estimated_var(infer): ", bn.moving_variance.numpy())
print("y(infer):", y_infer.numpy())

