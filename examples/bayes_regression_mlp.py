import autograd.numpy as np
import autograd.scipy.stats.norm as norm

import sys; sys.path.append('..')
from mcmc import langevin, RWMH, HMC, RK_langevin

from matplotlib import pyplot as plt
from copy import copy

import argparse
parser = argparse.ArgumentParser(description='Bayesian Regression')
parser.add_argument('--iters', type=int, default=2000,
					help='number of iteratons (default: 2000)')
parser.add_argument('--step-size', type=float, default=1e-3,
					help='learning rate (default: 0.001)')
parser.add_argument('--samples', type=int, default=1,
					help='Number of mcmc samples (default: 25)')
parser.add_argument('--alg', type=str, default='langevin',
					help="which MCMC algorithm to use. One of ['langevin', 'MALA', 'RK', 'RWMH', 'HMC']. Default:'langevin'")
parser.add_argument('--seed', type=int, default=None,
					help='int random seed (default: None, no seed set.)')
args = parser.parse_args()

if args.seed is not None: np.random.seed(args.seed)


#===== MLP Definition =========

def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
	"""These functions implement a standard multi-layer perceptron,
	vectorized over both training examples and weight samples."""
	shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
	num_weights = sum((m+1)*n for m, n in shapes)

	def unpack_layers(weights):
		num_weight_sets = len(weights)
		for m, n in shapes:
			yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
				  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
			weights = weights[:, (m+1)*n:]

	def predictions(weights, inputs):
		"""weights is shape (num_weight_samples x num_weights)
		   inputs  is shape (num_datapoints x D)"""
		inputs = np.expand_dims(inputs, 0)
		for W, b in unpack_layers(weights):
			outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
			inputs = nonlinearity(outputs)
		return outputs

	def logprob(weights, inputs, targets):
		log_prior = -L2_reg * np.sum(weights**2, axis=1)
		preds = predictions(weights, inputs)
		log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
		return log_prior + log_lik

	return num_weights, predictions, logprob


def build_dataset():
	n_train, n_test, d = 200, 100, 1
	xlims = [-1.0, 5.0]
	x_train = np.random.rand(n_train, d) * (xlims[1] - xlims[0]) + xlims[0]
	x_test = np.random.rand(n_test, d) * (xlims[1] - xlims[0]) + xlims[0]

	target_func = lambda t: (np.log(t + 100.0) * np.sin(1.0 * np.pi*t)) + 0.1 * t

	y_train = target_func(x_train)
	y_test = target_func(x_test)

	y_train += np.random.randn(*x_train.shape) * (1.0 * (x_train + 2.0)**0.5)


	return (x_train, y_train), (x_test, y_test)

def gaussian_conf_bands(data, alpha=0.68):
	"""
	Y columns are independent samples, 
	(X[i], Y[i]) is a set of Y.shape[1] samples at X[i])
	"""
	X,Y = data
	U,L = [],[]
	for y in Y:
		mu = np.mean(y)
		sigma = np.std(y) #no dof correction...
		L.append(mu - sigma)
		U.append(mu + sigma)

	return X, L, U

def main():
	# ======= Setup =======

	rbf = lambda x: np.exp(-x**2)#deep basis function model
	num_weights, predictions, logprob = \
		make_nn_funs(layer_sizes=[1, 128, 1], L2_reg=0.1,
					noise_variance=0.5, nonlinearity=np.tanh)

	(x_train, y_train), (x_test, y_test) = build_dataset()

	log_posterior = lambda weights: logprob(weights, x_train, y_train)

	sig = 5.0
	if args.alg in ['RWMH']: sig = 1.0
	init_weights = np.random.randn(args.samples, num_weights) * sig

	hist = []
	def callback(x,i,*args):
		hist.append(copy(log_posterior(x)))

	# ======= Sampling =======

	if args.alg == 'langevin':
		weight_samps = langevin(log_posterior, init_weights,  	
								num_iters = args.iters, num_samples = args.samples, 
								step_size = args.step_size, callback = callback)
	
	elif args.alg == 'RWMH': #doesn't work well
		weight_samps = RWMH(log_posterior, init_weights, 
							num_iters = args.iters, num_samples = args.samples, 
							sigma = 10 * args.step_size, callback = callback)
	
	elif args.alg == 'HMC':
		n_leap_iters = 5
		weight_samps = HMC(log_posterior, init_weights, 
							num_iters = args.iters//n_leap_iters, num_leap_iters = n_leap_iters,
							num_samples = args.samples, step_size = args.step_size, 
							callback = callback)

	elif args.alg == 'RK':
		weight_samps = RK_langevin(log_posterior, init_weights, 	
								num_iters = args.iters, num_samples = args.samples, 
								step_size = args.step_size, callback = callback)

	else:
		raise ValueError("Allowable mcmc algorithms are ['langevin', 'MALA', 'RK', 'RWMH', 'HMC'], got {}".format(args.alg))

	# ===== Plotting ======

	plot_inputs = np.linspace(-1, 6, num=400)
	outputs = predictions(weight_samps, np.expand_dims(plot_inputs, 1))

	_, lower, upper = gaussian_conf_bands((plot_inputs, outputs[:,:,0].T))

	f, axes = plt.subplots(2)

	hist = np.array(hist)
	axes[0].plot(hist, alpha=0.5)
	
	axes[1].plot(x_train.ravel(), y_train.ravel(), 'bx', color='green')
	axes[1].plot(x_test.ravel(), y_test.ravel(), 'bx', color='red')
	axes[1].plot(plot_inputs, outputs[:, :, 0].T, alpha=0.25)
	axes[1].plot(plot_inputs, np.mean(outputs[:10, :, 0].T, axis=1), color='black', linewidth=1.0)
	axes[1].fill_between(plot_inputs, lower, upper, alpha=0.75)


	plt.show()


if __name__=='__main__':
	main()

