import autograd.numpy as np
from autograd import elementwise_grad

from tqdm import tqdm
from copy import copy


def leapfrog(g, x, r, eps = 0.01, num_iters = 10):
	""" vectorized leapfrog integrator
	g is a callable that evaluates the elementwise grad with signature g(x,i)
	x is the starting point
	r is the staring momentum
	"""
	for i in range(num_iters):
		r += 0.5 * eps * g(x)
		x += eps * r
		r += 0.5 * eps * g(x)

	return x,r

def batch_dot_product(a, b):
	return np.sum(a * b, axis=1)

def HMC(logprob, x0, num_iters = 1000, num_leap_iters=10, step_size = 0.01, num_samples=10, callback = None):
	"""
	logprob:	autograd.np-valued function that accepts array x of shape x0.shape 
				and returns array of shape (x.shape[0],) representing log P(x)
				up to a constant factor.

		x0: 	inital array of inputs.
	"""
	assert x0.shape[0] == (num_samples), 'x0.shape[0] must be num_samples.'
	x = x0
	g = elementwise_grad(logprob)
	
	for i in tqdm(range(num_iters)):
		U = np.random.rand(num_samples, 1)
		R = - np.random.randn(*x.shape)
		xnew, rnew = leapfrog(g, x, R, step_size, num_iters=num_leap_iters)
		log_alpha = logprob(xnew) - 0.5*batch_dot_product(rnew,rnew) - \
					 (logprob(x) - 0.5 * batch_dot_product(R,R))
		idxs =  np.log(U).ravel() < log_alpha.ravel()
		if callback: callback(x, i, xnew, log_alpha)
		x[idxs] = xnew[idxs]

	return x

