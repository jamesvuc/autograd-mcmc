#1d_tests.py

import autograd.numpy as np
import autograd.scipy.stats.norm as norm

from copy import copy
import datetime as dt
from matplotlib import pyplot as plt
import seaborn as sns

import sys; sys.path.append('..')
from mcmc import langevin, MALA, RK_langevin, RWMH, HMC

def bimodal_logprob(z):
	return (np.log(np.sin(z)**2) + np.log(np.sin(2*z)**2) + norm.logpdf(z)).ravel()

def main():

	#====== Setup =======

	n_iters, n_samples = 2500, 500
	init_vals = np.random.randn(n_samples, 1)

	allsamps = []
	logprob = bimodal_logprob

	#====== Tests =======

	t = dt.datetime.now()
	print('running 1d tests ...')
	samps = langevin(logprob, copy(init_vals),
					 num_iters = n_iters, num_samples = n_samples, step_size = 0.05)
	print('done langevin in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

	samps = MALA(logprob, copy(init_vals),
					 num_iters = n_iters, num_samples = n_samples, step_size = 0.05)
	print('done MALA in', dt.datetime.now()-t,'\n')
	allsamps.append(samps)

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

	lims = [-5,5]
	names = ['langevin', 'MALA', 'langevin_RK', 'RW MH', 'HMC']
	f, axes = plt.subplots(len(names), sharex=True)
	for i, (name, samps) in enumerate(zip(names, allsamps)):

		sns.distplot(samps, bins=1000, kde=False, ax=axes[i])
		axb = axes[i].twinx()
		axb.scatter(samps, np.ones(len(samps)), alpha=0.1, marker='x', color='red')
		axb.set_yticks([])

		zs = np.linspace(*lims, num=250)
		axes[i].twinx().plot(zs, np.exp(bimodal_logprob(zs)), color='orange')

		axes[i].set_xlim(*lims)
		title = name
		axes[i].set_title(title)

	plt.show()


if __name__ == '__main__':
	main()