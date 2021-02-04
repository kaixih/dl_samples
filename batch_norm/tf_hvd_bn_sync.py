import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Make sure that different ranks have different inputs.
tf.random.set_seed(hvd.local_rank())

sync_bn = hvd.SyncBatchNormalization(axis=-1, momentum=0.5,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='ones', moving_variance_initializer='ones')

x0 = tf.random.uniform((2,1,2,3))

print("x:", x0.numpy())
y0 = sync_bn(x0, training=True)
print("moving_mean: ", sync_bn.moving_mean.numpy())
print("moving_var: ", sync_bn.moving_variance.numpy())
print("y:", y0.numpy())
