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
io_p0 = 0.0
io_pf = 10.0
io_tpts = 100
io_tmin = 0
io_tmax = 10
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
	if arg[i][0]=="-" and arg[i][1:3]=="tmax":
		io_tmax = float(arg[i+1])
	if arg[i][0]=="-" and arg[i][1:3]=="tmin":
		io_tmin = float(arg[i+1])
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
	
	
	io_B0 = 0.5
	io_w0 = 2
	io_wB = 3.981737268
	io_tmax = 13
	io_tpts = 400
	io_pf = 6.294
	
	
	bv.B0 = io_B0
	bv.p0 = io_p0
	bv.pf = io_pf
	bv.tmax = io_tmax
	bv.tmin = io_tmin
	bv.tpts = io_tpts
	bv.w0 = io_w0
	bv.wB = io_wB
	
	bv_res = bv.solve_me()
	
	
	bs = blochsphere()
	fig = bs.plot(bv_res[0],bv_res[1],bv_res[2],bv_res[3])
	plt.show()
		
	if save_str=="save_plot":
		bs = blochsphere()
		fig = bs.plot(bv_res[0],bv_res[1],bv_res[2],bv_res[3])
		plt.savefig("B0_pf_im_inset.png")
	elif save_str=="show":
		bs = blochsphere()
		fig = bs.plot(bv_res[0],bv_res[1],bv_res[2],bv_res[3])
		plt.show()
	elif save_str=="save_data":
		np.savetxt(
			"test.dat",
			bv_res,
			delimiter="\t",
			header="t\tsx\tsy\tsz",
			comments="#\n#"
		)
			

elif mode_str=="1dscan":
	par1min = 3.981737268-1
	par1max = 3.981737268+1
	par1pts = 1000

	par1 = np.linspace(par1min,par1max,par1pts)

	res = np.zeros((3,par1pts))

	perc = 0
	k = 0


	for i in range(par1pts):
		bv = blochvector()

		io_B0 = 0.5
		io_w0 = np.pi#2
		io_wB = 3.981737268
		io_tmax = 100
		io_tpts = 4000
		io_pf = 100

		
		bv.B0 = io_B0
		bv.p0 = io_p0
		bv.pf = io_pf
		bv.tmax = io_tmax
		bv.tmin = io_tmin
		bv.tpts = io_tpts
		bv.w0 = io_w0
		bv.wB = io_wB

		bv.wB = par1[i]

		bv_res = bv.solve_me()
		
		min_index = np.argmin(bv_res[3])
		
		res[0][i] = par1[i]
		res[1][i] = bv_res[3][min_index]
		res[2][i] = bv_res[0][min_index]
	
		if k/par1pts >= (perc+10)/100:
			perc+=10
			print(str(perc)+"%")
		k+=1

		
	if save_str=="save_plot":
		fig = plt.plot(res[0],res[1])
		plt.savefig("B0_pf_im_inset.png")
	elif save_str=="show":
		fig = plt.plot(res[0],res[1])
		plt.show()
	elif save_str=="save_data":
		fname = "1dscan"
		fname+=".dat" 
		
		
		np.savetxt(
			fname,
			np.asarray(list(zip(*res))),
			delimiter="\t",
			header="wb\tmin(sz)\tt",
			comments="#\n#"
		)
		
				
	


elif mode_str=="2dscan":
	par1min = 0.1
	par1max = 5
	par1pts = 10

	par1 = np.linspace(par1min,par1max,par1pts)

	par2min = 0.1
	par2max = 5
	par2pts = 10

	par2 = np.linspace(par2min,par2max,par2pts)

	res = np.zeros((3,par1pts*par2pts))

	k = 0
	perc = 0



	for i in range(par1pts):
		for j in range(par2pts):
			bv = blochvector()
			

			io_B0 = 0.5
			io_tmax = 100
			io_tpts = 400
			io_pf = 100

			bv.B0 = io_B0
			bv.pf = io_pf
			bv.tmax = io_tmax
			bv.tpts = io_tpts
			
			bv.w0 = par1[i]
			bv.wB = par2[j]
			
			bv_res = bv.solve_me()		
			
			res[0][k]=par1[i]
			res[1][k]=par2[j]

			#start_index=int(np.min(np.argwhere(bv_res[0]>bv.pf)))
			

			#res[2][k]=np.min(bv_res[3][start_index:])
			
			res[2][k]=np.min(bv_res[3])
			
			if k/(wBpts*pfpts) >= (perc+10)/100:
				perc+=10
				print(str(perc)+"%")
			k+=1


	if save_str=="save_plot":
		bs = blochsphere()
		fig = bs.plot(bv_res[0],bv_res[1],bv_res[2],bv_res[3])
		plt.savefig("B0_pf_im_inset.png")
	elif sava_str=="save_data":
		np.savetxt(
			"2dscan.dat",
			bv_res,
			delimiter="\t",
			header="t\tsx\tsy\tsz",
			comments="#\n#"
		)
		
	elif save_str=="show":
		bs = blochsphere()
		fig = bs.plot(bv_res[0],bv_res[1],bv_res[2],bv_res[3])
		plt.show()


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




