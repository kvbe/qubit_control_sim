import os
import numpy as np
import matplotlib.pyplot as plt



table = np.loadtxt("1dscan_resonant.dat.txt")

table = np.asarray(list(zip(*table)))


plt.plot(table[0],table[1],lw=1,label=r"$\omega_z=2$")




table = np.loadtxt("1dscan_offresonant.dat.txt")

table = np.asarray(list(zip(*table)))


plt.xlabel(r"$\Omega_b$")
plt.ylabel(r"$min(\langle\sigma_z\rangle)$")



plt.plot(table[0],table[1],lw=1,label=r"$\omega_z=\pi$")

plt.legend()








plt.show()
