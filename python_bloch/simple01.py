import numpy as np
import random as rn
import matplotlib.pyplot as plt

from qutip import *
from plot_bloch_sphere import blochsphere
from scipy.interpolate import griddata


class blochvector:
	def __init__(
		self,
		w0=1.0,
		B0=10.0,
		wB=1,
		p0=0.0,
		pf=np.pi/2,
		tmin=0,
		tmax=10,
		tpts=400		
	):
		self.w0=w0
		self.B0=B0
		self.wB=wB
		self.p0=p0
		self.pf=pf
		self.tmin=tmin
		self.tmax=tmax
		self.tpts=tpts
	
	def solve_me(self):
		def rect(x,start=0,end=1):
			y=x
			if type(x)==float:
				if x <= end and x >= start:
					y = 1
				else:
					y = 0
			else:
				for i in range(len(x)):
					if x[i] <= end and x[i] >= start:
						y[i] = 1
					else:
						y[i] = 0
			return y		
		

		w0=self.w0
		B0=self.B0
		wB=self.wB
		p0=self.p0
		pf=self.pf
		
		tmin=self.tmin
		tmax=self.tmax
		tpts=self.tpts
		
		t = np.linspace(tmin,tmax,tpts)

		psi0 = tensor(basis(2,0))


		H0 = 1/2*w0*sigmaz()
		H1 = 1/2*B0*sigmax()
		
		c_op_list = []


		def H1_coeff(t, args):
			return np.cos(wB*t)*rect(t,start=p0,end=pf)

		H = [H0,[H1, H1_coeff]]



		output = mesolve(H, psi0, t, c_op_list, [sigmax(), sigmay(), sigmaz()])

		self.t = t
		self.sx = output.expect[0]
		self.sy = output.expect[1]
		self.sz = output.expect[2]
		
		return [t,self.sx,self.sy,self.sz]


		def plot(self, save="False"):
			bs = blochsphere()
			plt.close()

			fig = bs.plot(self.t,self.sx,self.sy,self.sz)

			plt.show()




bv = blochvector()


bv.B0=10
bv.wB=0
bv.p0=0
bv.pf=np.pi


bv_res = bv.solve_me()

bs = blochsphere()

bs.plot(bv_res[0],bv_res[1],bv_res[2],bv_res[3])



'''
wBmin=5.421-0.4
wBmax=5.421+0.4
wBpts=10

wB=np.linspace(wBmin,wBmax,wBpts)

pfmin=1.813-0.4
pfmax=30
pfpts=30

pf=np.linspace(pfmin,pfmax,pfpts)

res = np.zeros((3,wBpts*pfpts))

bv = blochvector()

bv.B0 = 40

k = 0
perc = 0



for i in range(wBpts):
	bv.wB=wB[i]
	for j in range(pfpts):
		bv.pf=pf[j]
		bv_res = bv.solve_me()		
		res[0][k]=wB[i]
		res[1][k]=pf[j]

		start_index=int(np.min(np.argwhere(bv_res[0]>pf[j])))
		

		res[2][k]=np.min(bv_res[3][start_index:])
		
			
		if k/(wBpts*pfpts) >= (perc+10)/100:
			perc+=10
			print(str(perc)+"%")
		k+=1





x = res[0]
y = res[1]
z = res[2]







grid_x, grid_y = np.meshgrid(wB,pf)



points = np.array(list(zip(x,y)))


gridded = griddata(points, res[2], (grid_x, grid_y), method='linear')

indeces_scat = np.argwhere(gridded == np.min(gridded))

gridded = np.flip(gridded, axis=0)



fig = plt.imshow(gridded, extent=(wBmin,wBmax,pfmin,pfmax), interpolation="nearest", aspect="auto",cmap="winter")








indeces_scat = np.array(list(zip(*indeces_scat)))
xi_scat = indeces_scat[0]
yi_scat = indeces_scat[1]




plt.scatter(
	grid_x[xi_scat,yi_scat],
	grid_y[xi_scat,yi_scat],
	s=20,
	c='red',
	marker="D"
)


print(grid_x[xi_scat,yi_scat])
print(grid_y[xi_scat,yi_scat])

#plt.scatter(x_scat,y_scat)

#x[min_index],y[min_index],z[min_index]

plt.colorbar(fig, label=r"$\mathrm{min}(\langle\sigma_z\rangle)$")

plt.xlabel(r"$B_0$")
plt.ylabel(r"$p_f$")


'''

plt.show()

#plt.savefig("B0_pf_im_inset.png")


