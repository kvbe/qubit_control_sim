import os
import numpy as np
import matplotlib.pyplot as plt



table = np.loadtxt("1dscan_resonant.dat")

table = np.asarray(list(zip(*table)))


plt.plot(table[0],table[1],lw=1)




table = np.loadtxt("1dscan_offresonant.dat")

table = np.asarray(list(zip(*table)))


plt.plot(table[0],table[1],lw=1)










plt.show()
