import numpy as np
import tensorflow as tf

# 使用Numpy模块实现hardmax
a = np.array([1, 2, 3, 4, 5]) # 创建ndarray数组
a_max_np = np.max(a)
print(a_max_np) # 5

# 使用TensorFlow深度学习框架实现hardmax：
print(tf.__version__) # 2.3.0
a_max_tf = tf.reduce_max(a)
print(a_max_tf) # tf.Tensor(5, shape=(), dtype=int32)