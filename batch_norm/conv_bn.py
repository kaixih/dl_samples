import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization

x_shape = (2,3,3,8)
conv = Conv2D(filters=4, kernel_size=(2, 2), padding='same', use_bias=False)
bn = BatchNormalization()

model = tf.keras.Sequential()
model.add(conv)
model.add(bn)
model.build(x_shape)

# Set the weights to fixed numbers to make the results deterministic.
conv_kernel = np.array(
    [[[[ 0.28928265,  0.23558983, -0.19925436,  0.279291  ],
       [-0.1196335 , -0.32112214,  0.01127437,  0.32169685],
       [-0.08317402,  0.16047171, -0.12463328,  0.23925844],
       [ 0.08999312, -0.03824228,  0.2053617 ,  0.2888197 ],
       [ 0.27664837, -0.12455405, -0.22641279,  0.04141778],
       [-0.08758721, -0.16842201, -0.02454785, -0.11628433],
       [-0.00394386, -0.2807856 , -0.21075574,  0.3088759 ],
       [ 0.27292117,  0.02746439,  0.1410568 ,  0.17971316]],

      [[-0.3313501 ,  0.28208616,  0.06316379,  0.21221998],
       [ 0.10167867,  0.19540873, -0.05865332, -0.19748352],
       [ 0.09938025, -0.06422496,  0.04848725,  0.0381521 ],
       [-0.22118548, -0.17929943,  0.14221469,  0.03561965],
       [ 0.10845953,  0.14317748,  0.31781027,  0.28947517],
       [ 0.14699289, -0.03167731,  0.32531878, -0.19360593],
       [ 0.04401925, -0.15699245, -0.23643766, -0.04881379],
       [-0.23063244, -0.26502782,  0.25755385,  0.04906145]]],


     [[[ 0.27046743, -0.26585987,  0.27125904, -0.30494314],
       [ 0.09588316, -0.27299273, -0.33849236, -0.0620313 ],
       [ 0.03550586,  0.08056217, -0.16947088,  0.24737474],
       [ 0.01032919, -0.22233675,  0.08465979, -0.3530779 ],
       [-0.22524515, -0.31767225, -0.34700757, -0.3306819 ],
       [-0.13729107, -0.2793208 ,  0.25541648,  0.33059844],
       [-0.09040532, -0.32072455,  0.27621415, -0.12335801],
       [-0.2558499 ,  0.11504766, -0.31129158,  0.18582192]],

      [[ 0.27576742, -0.14861712,  0.14516303,  0.07628378],
       [-0.25931808,  0.28388456,  0.11597389,  0.04469427],
       [ 0.18877807, -0.12930459, -0.26642826, -0.09548005],
       [-0.29668954, -0.0793331 , -0.12217039,  0.07592654],
       [-0.26208022, -0.029412  ,  0.0556823 , -0.11517233],
       [-0.3197474 ,  0.35219547, -0.34922153, -0.2841328 ],
       [ 0.02157903,  0.32386252, -0.20609969,  0.20551983],
       [ 0.19358537, -0.11206484,  0.06496733, -0.33009326]]]])
bn_gamma = np.array([0.27576742, -0.14861712,  0.14516303,  0.07628378])
bn_beta = np.array([0.18877807, -0.12930459, -0.26642826, -0.09548005])
bn_moving_mean = np.array([0., 0., 0., 0.])
bn_moving_variance = np.array([1., 1., 1., 1.])

model.layers[0].set_weights([conv_kernel])
model.layers[1].set_weights([bn_gamma, bn_beta, bn_moving_mean,
                             bn_moving_variance])

x = tf.ones(x_shape)

y = model(x)
print(y)
print("X shape:", x.shape)
print("Conv kernel shape:", conv.kernel.shape)
print("BN gamma shape:", bn.gamma.shape)
print("BN beta shape:", bn.beta.shape)