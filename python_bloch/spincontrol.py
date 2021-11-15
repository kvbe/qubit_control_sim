import sys
import numpy as np
import random as rn
import matplotlib.pyplot as plt

from qutip import *
from blochsphere import blochsphere
from blochvector import blochvector
from scipy.interpolate import griddata 







mode_str = "single"
save_str = "save"
plot_str = "bgc"

io_B0 = 1.0
io_p0 = 1.0
io_pf = 2.0
io_tpts = 100
io_w0 = 1.0
io_wB = 1.0

sweep_args = ""

arg = sys.argv

for i in range(len(arg)):
	#modes
	if arg[i][0]=="m" and arg[i][1]==":":
		mode_str = arg[i][2:]
	if arg[i][0]=="s" and arg[i][1]==":":
		save_str = arg[i][2:]
	if arg[i][0]=="p" and arg[i][1]==":":
		plot_str = arg[i][2:]
	
	#parameters
	if arg[i][0]=="-" and arg[i][1:3]=="B0":
		io_B0 = float(arg[i+1])
	if arg[i][0]=="-" and arg[i][1:3]=="p0":
		io_p0 = float(arg[i+1])
	if arg[i][0]=="-" and arg[i][1:3]=="pf":
		io_pf = float(arg[i+1])
	if arg[i][0]=="-" and arg[i][1:5]=="tpts":
		io_tpts = int(arg[i+1])
	if arg[i][0]=="-" and arg[i][1:3]=="w0":
		io_w0 = float(arg[i+1])
	if arg[i][0]=="-" and arg[i][1:3]=="wB":
		io_wB = float(arg[i+1])
	
	#sweep arguments
	if arg[i][0:8]=="m:sweep[":
		sweep_args = arg[i][8:-1]
		
	
	







if mode_str=="single":
	bv = blochvector()
	
	bv.B0 = io_B0
	bv.p0 = io_p0
	bv.pf = io_pf
	bv.w0 = io_w0
	bv.wB = io_wB
	
	bv_res = bv.solve_me()
	bs = blochsphere()
	
	fig = bs.plot(bv_res[0],bv_res[1],bv_res[2],bv_res[3])

elif mode_str=="sweep":
	par1min = 0
	par1max = 1
	par1pts = 10

	par1 = np.linspace(par1min,par1max,par1pts)

	par2min = np.pi
	par2max = 30
	par2pts = 10

	par2 = np.linspace(par2min,par2max,par2pts)

	res = np.zeros((3,par1pts*par2pts))

	bv = blochvector()



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



if save_str=="save":
	plt.savefig("B0_pf_im_inset.png")
elif save_str=="show":
	plt.show()
