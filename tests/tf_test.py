import tensorflow as tf
from tensorflow.python.client import device_lib

print(tf.__version__)

# tf v1.x
print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# print(device_lib.list_local_devices())

# tf v2.x
# print(tf.config.list_physical_devices('GPU'))
