
import autograd.numpy as np
from autograd import elementwise_grad

from copy import copy
from tqdm import tqdm


def RWMH(logprob, x0, num_iters=1000, num_samples =10, sigma = 0.5, callback=None):
	"""
	logprob:	autograd.np-valued function that accepts array x of shape x0.shape 
				and returns array of shape (x.shape[0],) representing log P(x)
				up to a constant factor.

		x0: 	inital array of inputs.
	"""
	assert x0.shape[0] == (num_samples), 'x0.shape[0] must be num_samples.'
	x = x0
	for i in tqdm(range(num_iters)):
		Z = np.random.randn(*x.shape) * sigma
		U = np.random.rand(num_samples)
		x_new = x + Z
		log_alpha = logprob(x_new).ravel() - logprob(x).ravel()
		idxs = np.log(U).ravel() < log_alpha
		if callback: callback(x, i, x_new, log_alpha)
		x[idxs] = x_new[idxs]

	return x