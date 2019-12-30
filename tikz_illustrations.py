import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

# plt.style.use("ggplot")

plt.plot([-3, -1, 3], [0, 0, 3], 'k', lw=0.6)
plt.plot([-3, 0, 3], [0, 0, 4], 'k', lw=0.7)
plt.plot([-3, 1, 3], [0, 0, 5], 'k', lw=0.8)

plt.plot([-3, -1, 1.285714, 2.142857, 3], [0, 0, 1.714286, 2.857143, 5], 'g.--', lw=2, ms=7)

# plt.grid(True)

# plt.show()

print(
    tikzplotlib.get_tikz_code()
)

