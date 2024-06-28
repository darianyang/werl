
import numpy as np
import matplotlib.pyplot as plt

#weights = np.load("op_weights.npy")
weights = np.loadtxt("op_weights.txt")
print(weights.shape)
#weights = np.load("op_weights-reap2-25i.npy")

plt.plot(weights[:,0], label="X")
plt.plot(weights[:,1], label="Y")
plt.legend()
plt.xlabel("WE Iteration")
plt.ylabel("Weight")
plt.ylim(0,1)
plt.show()
