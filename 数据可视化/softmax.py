import tensorflow as tf

# # 使用TensorFlow深度学习框架实现softmax
# print(tf.__version__)  # 2.3.0
# a = tf.constant([2, 3, 5], dtype=tf.float32)
#
# b1 = a / tf.reduce_sum(a)  # 不使用指数
# print(b1)  # tf.Tensor([0.2 0.3 0.5], shape=(3,), dtype=float32)
#
# b2 = tf.nn.softmax(a)  # 使用指数的Softmax
# print(b2)  # tf.Tensor([0.04201007 0.11419519 0.8437947 ], shape=(3,), dtype=float32)
#
# import numpy as np
#
# # 使用Numpy模块实现softmax
# scores = np.array([123, 456, 789])
# softmax = np.exp(scores) / np.sum(np.exp(scores))
# print(softmax)  # [ 0.  0. nan]
#
# # 针对数值溢出有其对应的优化方法，将每一个输出值减去输出值中最大的值。
# scores = np.array([123, 456, 789])
# scores -= np.max(scores)
# p = np.exp(scores) / np.sum(np.exp(scores))
# print(p) # [5.75274406e-290 2.39848787e-145 1.00000000e+000]

import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        # Matrix
        # shift max whithin each row
        constant_shift = np.max(x, axis=1).reshape(1, -1)
        x -= constant_shift
        x = np.exp(x)
        normlize = np.sum(x, axis=1).reshape(1, -1)
        x /= normlize
    else:
        # vector
        constant_shift = np.max(x)
        x -= constant_shift
        x = np.exp(x)
        normlize = np.sum(x)
        x /= normlize
    assert x.shape == orig_shape
    return x


softmax_inputs = np.arange(-10, 10, 0.1)
softmax_outputs = softmax(softmax_inputs)
print("Sigmoid Function Input :: {}".format(softmax_inputs))
print("Sigmoid Function Output :: {}".format(softmax_outputs))
# 画图像
plt.plot(softmax_inputs, softmax_outputs)
plt.xlabel("Softmax Inputs")
plt.ylabel("Softmax Outputs")
plt.show()
