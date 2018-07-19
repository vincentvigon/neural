import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()
ax.set_xlim(( 0, 2))
ax.set_ylim((-2, 2))

line, = ax.plot([],[])

x = np.linspace(0, 2, 1000)
y = np.sin(2 * np.pi * x)
line.set_data(x, y)

plt.show()