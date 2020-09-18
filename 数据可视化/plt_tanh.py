import matplotlib.pyplot as plt
import numpy as np


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

x = np.linspace(-10, 10)
y = tanh(x)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.set_yticks([-1, -0.5, 0.5, 1])

plt.plot(x, y, label="Tanh", color="red")
plt.legend()
plt.show()
