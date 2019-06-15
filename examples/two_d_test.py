import autograd.numpy as np
import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.multivariate_normal as mvn

from copy import copy
import datetime as dt

from matplotlib import pyplot as plt
import seaborn as sns

import sys; sys.path.append('..')
from mcmc import langevin, MALA, RK_langevin, RWMH, HMC


def make_logprob():

	mus = np.array([[0.0, 0.0],
					[2.5,  4.0],
					[4.0,  0.0]])

	sigmas = np.array([	[[1.0, 0.0],
					 	 [0.0, 2.0]],
						
						[[2.0, -1.0],
					  	 [-1.0, 1.0]],

						[[1.0, 0.1],
						 [0.1, 2.0]] ])

	def _logprob(z):
		return np.log(mvn.pdf(z, mean=mus[0], cov=sigmas[0]) +\
			   		  mvn.pdf(z, mean=mus[1], cov=sigmas[1]) +\
			   		  mvn.pdf(z, mean=mus[2], cov=sigmas[2]))

	return _logprob


def main():
	#====== Setup =======
	n_iters, n_samples, d = 2500, 2000, 2
	init_vals = np.random.rand(n_samples, d) * 5.0

	logprob = make_logprob()
	allsamps = []

	#====== Tests =======

	t = dt.datetime.now()
	print('running 2d tests ...')
	samps = langevin(logprob, copy(init_vals),
					 num_iters = n_iters, num_samples = n_samples, step_size = 0.05)
	print('done langevin in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	samps = MALA(logprob, copy(init_vals),
					 num_iters = n_iters, num_samples = n_samples, step_size = 0.05)
	print('done MALA in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	t = dt.datetime.now()
	samps = RK_langevin(logprob, copy(init_vals),
				 num_iters = n_iters, num_samples = n_samples, step_size = 0.01)
	print('done langevin_RK in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	t = dt.datetime.now()
	samps = RWMH(logprob, copy(init_vals), 
				  num_iters = n_iters, num_samples = n_samples, sigma = 0.5)
	print('done RW MH in' , dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	t = dt.datetime.now()
	samps = HMC(logprob, copy(init_vals),
				num_iters = n_iters//5, num_samples = n_samples, 
				step_size = 0.05, num_leap_iters=5)
	print('done HMC in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	#====== Plotting =======

	pts = np.linspace(-7, 7, 1000)
	X, Y = np.meshgrid(pts, pts)
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X
	pos[:, :, 1] = Y
	Z = np.exp(logprob(pos))


	f, axes = plt.subplots(2,3)
	names = ['langevin', 'MALA', 'langevin_RK', 'RW MH', 'HMC']
	for i, (name, samps) in enumerate(zip(names, allsamps)):

		row = i // 3
		col = i % 3
		ax = axes[row, col]

		ax.contour(X, Y, Z)
		ax.hist2d(samps[:,0], samps[:,1], alpha=0.5, bins=25)
		ax.set_title(name)
	

	axes[1,2].hist2d(init_vals[:,0], init_vals[:,1], bins=25)
	axes[1,2].set_title('Initial samples.')

	plt.show()




if __name__ == '__main__':
	main()

	



