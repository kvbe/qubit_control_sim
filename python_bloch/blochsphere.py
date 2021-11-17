import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection

class blochsphere:

	def __init__(
		self,
		surf_pts=25,
		surf_alpha=0.25,
		wf_alpha=0.5,
		wf_lw=0.5
	):
		self.surf_pts = surf_pts
		self.surf_alpha = surf_alpha
		
		self.wf_alpha = wf_alpha
		self.wf_lw = wf_lw
		
		self.fig = plt.figure(figsize=plt.figaspect(0.5))



	def plot(
		self,
		t_in,
		x_in,
		y_in,
		z_in,
		g_in=False
	):
		surf_pts=self.surf_pts
		surf_alpha=self.surf_alpha
		wf_alpha=self.wf_alpha
		wf_lw=self.wf_lw
		
		fig = plt.figure(figsize=(16, 5))

		axl = fig.add_subplot(1, 2, 1, projection='3d')

		U = np.linspace(0, np.pi, surf_pts)
		V = np.linspace(0, np.pi, surf_pts)
		U, V = np.meshgrid(U, V)
		X = np.cos(U)*np.sin(V)
		Y = np.sin(U)*np.sin(V)
		Z = np.cos(V)

		axl.plot_surface(
			X, Y, Z,
			color='#FFDDDD',
			alpha=surf_alpha,
			linewidth=0,
			antialiased=True
		)

		axl.plot_wireframe(
			X, Y, Z,
			rstride=int(surf_pts/4),
			cstride=int(surf_pts/4),
			linewidth=wf_lw,
			color="grey",
			alpha=wf_alpha
		)


		axl.plot(
			[-1,1],[0,0],[0,0],
			linewidth=wf_lw,
			color="grey",
			alpha=wf_alpha
		)

		axl.plot(
			[0,0],[-1,1],[0,0],
			linewidth=wf_lw,
			color="grey",
			alpha=wf_alpha
		)

		axl.plot(
			[0,0],[0,0],[-1,1],
			linewidth=wf_lw,
			color="grey",
			alpha=wf_alpha
		)

		U = np.linspace(np.pi, 2*np.pi, surf_pts)
		V = np.linspace(0, np.pi, surf_pts)
		U, V = np.meshgrid(U, V)
		X = np.cos(U)*np.sin(V)
		Y = np.sin(U)*np.sin(V)
		Z = np.cos(V)

		axl.plot_surface(
			X, Y, Z,
			color='#FFDDDD',
			alpha=surf_alpha,
			linewidth=0,
			antialiased=True
		)

		axl.plot_wireframe(
			X, Y, Z,
			rstride=int(surf_pts/4),
			cstride=int(surf_pts/4),
			linewidth=wf_lw,
			color="grey",
			alpha=wf_alpha
		)


		# turn axis off
		axl.set_axis_off()
		# make the panes transparent
		axl.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		axl.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		axl.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		# make the grid lines transparent
		axl.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		axl.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		axl.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)


		axl.set_xlim(-1.3, 1.3)
		axl.set_ylim(-1.3, 1.3)
		axl.set_zlim(-1, 1)

		axl.text(1.05, 0, -0.05, r"$x$", color='black')
		axl.text(0, 1.05, -0.05, r"$y$", color='black')

		axl.text(0, 0, 1.1, r"$\vert 0 \rangle $", color='black')
		axl.text(0, 0, -1.15, r"$\vert 1 \rangle $", color='black')



		scat = axl.scatter(x_in, y_in, z_in, c=t_in, cmap="seismic")

		
		axr = fig.add_subplot(1, 2, 2)
		
		axr.plot(t_in, x_in, label=r"$\langle\sigma_x\rangle$")
		axr.plot(t_in, y_in, label=r"$\langle\sigma_y\rangle$")
		axr.plot(t_in, z_in, label=r"$\langle\sigma_z\rangle$")

		axr.legend()
		axr.set_xlabel(r'$t$')
		
		'''
		if g_in:
			axrr = fig.add_subplot(1, 3, 3)
			axrr.plot(t_in, x_in, label=r"$\langle\sigma_x\rangle$")
		'''
		
		
		cbar= fig.colorbar(
			scat,
			ax=axl,
			#aspect=1,
			shrink=0.5,
			location='left'
		)
		cbar.set_label(r'$t$', rotation=0)

		

		
		return fig













