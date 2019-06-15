
import autograd.numpy as np
from autograd import elementwise_grad

from copy import copy
from tqdm import tqdm


def langevin(logprob, x0, num_iters = 1000, num_samples=10, step_size = 0.01, callback = None):
	"""
	logprob:	autograd.np-valued function that accepts array x of shape x0.shape 
				and returns array of shape (x.shape[0],) representing log P(x)
				up to a constant factor.
		x0: 	inital array of inputs.
	"""
	assert x0.shape[0] == num_samples, 'x0.shape[0] must be num_samples.'
	x = x0
	g = elementwise_grad(logprob)
	
	for i in tqdm(range(num_iters)):
		Z = np.random.randn(*x.shape)
		dx = g(x)
		if callback: callback(x,i,dx)
		x += 0.5*(step_size**2)*dx + step_size*Z

	return x


def MALA(logprob, x0, num_iters = 1000, num_samples=10, step_size = 0.01, callback = None):
	"""
	logprob:	autograd.np-valued function that accepts array x of shape x0.shape 
				and returns array of shape (x.shape[0],) representing log P(x)
				up to a constant factor.

		x0: 	inital array of inputs.
	"""
	assert x0.shape[0] == num_samples, 'x0.shape[0] must be num_samples.'
	x = x0
	g = elementwise_grad(logprob)
	logq = lambda y, x: - 0.5 * np.sum(((y - x - 0.5*step_size**2 * g(x))  / step_size)**2, axis=1)#log q(y|x)
	
	for i in tqdm(range(num_iters)):
		Z = np.random.randn(*x.shape)
		U = np.random.rand(num_samples)
		dx = g(x)
		x_new = x + 0.5*(step_size**2)*dx + step_size * Z
		log_alpha = (logprob(x_new).ravel() + logq(x, x_new) 
					- logprob(x).ravel() - logq(x_new, x))
		idxs = np.log(U) < log_alpha
		if callback: callback(x, i, x_new, log_alpha)

		x[idxs] = x_new[idxs]

	return x


def srk2(a, b, x0, num_iters = 1000, num_samples=10, step_size = 0.01, callback = None):
	""" A generic stochastic 2nd-order Runge-Kutta solver for homogeneous diffusions of the form
			dX_t = a(X_t)dt + b(X_t)dW
		where a() and b() are callable, sample-vectorized functions of X_t. """
	assert x0.shape[0] == num_samples, 'x0.shape[0] must be num_samples.'
	x = x0
	h = step_size
	root_h = np.sqrt(h)

	for i in tqdm(range(num_iters)):
		W = np.sqrt(h) * np.random.randn(*x.shape)
		S = np.random.randint(0, 1, size = (num_samples, 1))

		K1 = h * a(x) + (W - root_h*S) * b(x)
		K2 = h * a(x + K1) + (W + root_h*S) * b(x+K1)

		if callback: callback(x, i)

		x += 0.5 * (K1 + K2)

	return x

def RK_langevin(logprob, x0, **kwargs):
	""" Treats langevin-based MCMC as the problem of numerically integrating
		the continuous-time SDE dX_t = 0.5 g(X_t)dt + dB_t where g(x) = grad(logprob(x)).
		This diffusion is the continuous-time langevin equation (aka Smoluchowski) equation,
		and has invariant measure proportional to exp(logprob(x)). """

	g = elementwise_grad(logprob)
	a = lambda x : 0.5 * g(x)
	b = lambda x: 1.0

	return srk2(a, b, x0, **kwargs)