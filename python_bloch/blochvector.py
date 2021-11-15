import numpy as np
import random as rn
import matplotlib.pyplot as plt

from qutip import *
from blochsphere import blochsphere
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

			fig = bs.plot(self.t,self.sx,self.sy,self.sz)
			
			return fig




