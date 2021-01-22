import numpy as np
import matplotlib.pyplot as plt

x = np.load('test2.npy', allow_pickle=True)
y = np.zeros((4096, 192), dtype=np.int16)

y[:, :32] = x[:4096, :]
y[:, 32:64] = x[3*4096:4*4096, :]
y[:, 64:96] = x[1*4096:2*4096, :]
y[:, 96:128] = x[4*4096:5*4096, :]
y[:, 128:160] = x[2*4096:3*4096, :]
y[:, 160:192] = x[5*4096:6*4096, :]

fig, ax = plt.subplots()

fig.set_size_inches((7, 7))
ax.imshow(y)
ax.set_aspect('auto')
fig.show()