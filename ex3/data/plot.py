import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt("2D_dataset.txt", delimiter=',')

plt.plot(data[:,0], data[:,1], "x")
plt.xlabel("$x^1$")
plt.ylabel("$x^2$")
plt.savefig("figures/data_2D.pdf")
