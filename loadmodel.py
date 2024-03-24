import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Verify if TensorFlow can access the GPU
# print("Is GPU available:", tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
    print("Your GPU is detected: ", tf.config.list_physical_devices('GPU'))
else:
    print("GPU not detected")

# Test cuDNN
try:
    with tf.device('/gpu:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    print("cuDNN is available and working:", c)
except RuntimeError as e:
    print("cuDNN is not available:", e)
