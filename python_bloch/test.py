import numpy as np
import matplotlib.pyplot as plt

xmin = 1
xmax = 16
x = np.arange(xmin,xmax) 


nmax = 40
n = np.arange(0,nmax)


res = np.zeros((len(x),nmax))



for i in range(len(x)):
	y = x[i]
	for j in range(nmax):
		res[i][j]=y
		if y == 1:
			y=y
		else:
			if y%2==0:
				y=int(y/2)
			else:
				y=3*y+1
		
		


for i in range(len(x)):
	plt.plot(n,res[i], label=r'$x_0=$'+str(x[i]))

plt.legend()
plt.xlabel('i')
plt.ylabel('x')
plt.show()
