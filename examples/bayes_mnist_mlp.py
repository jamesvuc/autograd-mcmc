# bayes_mnsit_mlp.py

from __future__ import absolute_import, division
from __future__ import print_function

import autograd.numpy as np
from autograd.scipy.misc import logsumexp

from mnist_utils import Mnist, Batches
from tensorflow import keras

import sys; sys.path.append('..')
from mcmc import langevin, RWMH, HMC

from matplotlib import pyplot as plt
from copy import copy

import argparse
parser = argparse.ArgumentParser(description='Deep Bayesian inference of a MLP on MNIST.')
parser.add_argument('--batch-size', type=int, default=512,
					help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=2,
					help='number of epochs to train (default: 2)')
parser.add_argument('--step-size', type=float, default=1e-3,
					help='learning rate (default: 0.001)')
parser.add_argument('--samples', type=int, default=10,
					help='Number of mcmc samples (default: 10)')
parser.add_argument('--alg', type=str, default='langevin',
					help="which MCMC algorithm to use. One of ['langevin', 'RWMH', 'HMC']. Default:'langevin'")
parser.add_argument('--seed', type=int, default=0,
					help='random seed (default: 0)')
args = parser.parse_args()
np.random.seed(args.seed)

"""
=================
ANN functions here
==================
"""
def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
	"""These functions implement a standard multi-layer perceptron,
	vectorized over both training examples and weight samples."""
	shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
	num_weights = sum((m+1)*n for m, n in shapes)

	def unpack_layers(weights):
		num_weight_samples = len(weights)
		for m, n in shapes:
			yield weights[:, :m*n]     .reshape((num_weight_samples, m, n)),\
				  weights[:, m*n:m*n+n].reshape((num_weight_samples, 1, n))
			weights = weights[:, (m+1)*n:]

	def predictions(weights, inputs):
		"""weights is shape (num_weight_samples x num_weights)
		   inputs  is shape (num_datapoints x D)"""
		inputs = np.expand_dims(inputs, 0) #needs to broadcast over all weights
		for W, b in unpack_layers(weights):
			outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
			inputs = nonlinearity(outputs)

		#log-softmax for classification
		return outputs - logsumexp(outputs, axis=2, keepdims=True)

	def logprob(weights, inputs, targets):
		log_prior = -L2_reg * np.sum(weights**2, axis=1)
		preds = predictions(weights, inputs)

		log_lik = np.sum(preds * targets, axis=(1,2))

		return log_prior + log_lik
	
	return num_weights, predictions, logprob

def accuracy(net, weights, inputs, targets):
	target_class = np.argmax(targets, axis=1)
	avg_logprobs = np.mean(net(weights, inputs), axis=0) #this is 'integrating' out the weights
	predicted_class = np.argmax(avg_logprobs, axis=1)
	return np.mean(predicted_class == target_class)

def batch_entropy(p):
	""" p: an array of (batch_size x n_classes) with P(class == i) in dimension 1 for each batch.
		Computes - sum(p * log(p)) taking care of zero probabilities. """
	p += 1e-12
	return - np.sum(p * np.log(p), axis=1, keepdims=True) #or could use p + 1e-12

def prediction_entropy(net, weights, inputs):
	""" computes entropy of the prediction distribution returned by 'net'
		on 'inputs' using parameters 'weights' in a batch-wise. 

		This is discrete-entropy, which is easier to estimate. The sampled weights 
		define samples from the posterior predictive distribution P(y == c|x, W) over 
		the classes c=0,...,C-1. We compute the entropy of this distribution for each
		sample in the batch of inputs. These can be subsequently averaged. """

	logprobs = net(weights, inputs) # n_samples x batch_size x n_classes
	predicted_class = np.argmax(logprobs, axis=-1) # n_samples x batch_size
	_n_classes = logprobs.shape[-1]
	_n_samples, _bsize = predicted_class.shape

	counts = np.zeros((_bsize, _n_classes))
	for i in range(_n_classes):
		idxs = predicted_class == i
		counts[:, i] = np.sum(idxs, axis=0)

	counts /= np.sum(counts, axis=1, keepdims=True)
	return batch_entropy(counts)

def main():

	""" init network """
	relu = lambda x: np.maximum(x, 0.)
	# layer_sizes = [784, 10, 10] #small, for testing
	# layer_sizes = [784, 1024, 10] #wide
	layer_sizes = [784, 100, 10] #~90% accuracy within 10 minutes on 10 samples

	num_weights, predictions, logprob = \
		make_nn_funs(layer_sizes=layer_sizes, L2_reg=1.0,
					noise_variance=0.01, nonlinearity=relu)
	init_scale = 1.0
	init_weights = np.random.randn(args.samples, num_weights) * init_scale + 0.0
	print('params / sample:{}'.format(num_weights))


	""" init dataset """
	print("Loading training data...")
	mnist = Mnist()
	(train_images, train_labels), (test_images,  test_labels) = mnist.load_data()
	print('Loaded.')
	
	train_images = train_images.reshape(train_images.shape[0], 28*28)/255.0
	test_images = test_images.reshape(test_images.shape[0], 28*28)/255.0

	train_labels=keras.utils.to_categorical(train_labels, 10)
	test_labels=keras.utils.to_categorical(test_labels, 10)

	val_split = 500
	val_images, train_images = train_images[:val_split], train_images[val_split:]
	val_labels, train_labels = train_labels[:val_split], train_labels[val_split:]
	
	train_batches = Batches(train_images, train_labels, batch_size = args.batch_size)
	num_train_batches = train_batches.num_batches


	""" do learning """
	n_iters = num_train_batches * args.epochs
	log_posterior = lambda weights: logprob(weights, *train_batches.next())

	hist = dict(iters=[], losses=[], acc=[], ent=[])
	def callback(x, i, *args):
		if i % 2 == 0:
			hist['iters'].append(i)
			hist['losses'].append(logprob(x, val_images, val_labels))
			hist['acc'].append(accuracy(predictions, x, val_images, val_labels))
			# hist['ent'].append(np.mean(prediction_entropy(predictions, x, val_images).ravel()))


	print('training...')

	if args.alg == 'langevin':
		weight_samps = langevin(log_posterior, init_weights, 
								num_iters = n_iters, num_samples = args.samples, 
								step_size = args.step_size, callback = callback)
	
	elif args.alg == 'RWMH': #doesn't work well
		weight_samps = RWMH(log_posterior, init_weights, 
							num_iters = n_iters, num_samples = args.samples, 
							sigma = 10 * args.step_size, callback = callback)
	
	elif args.alg == 'HMC':
		n_leap_iters = 5
		weight_samps = HMC(log_posterior, init_weights, 
								num_iters = n_iters//n_leap_iters, num_leap_iters = n_leap_iters,
								num_samples = args.samples, step_size = args.step_size, 
								callback = callback)

	else:
		raise ValueError("Allowable mcmc algorithms are ['langevin', 'RWMH', 'HMC'], got {}.".format(args.alg))

	#=== evaluation ====

	acc = accuracy(predictions, weight_samps, test_images, test_labels)
	ent = prediction_entropy(predictions, weight_samps, test_images)

	n_classes = 10.0
	print('test accuracy:', acc)
	print('average test entropy=', np.mean(ent.ravel()))
	print('theoretical max entropy is', np.log(n_classes))

	all_losses = np.vstack(hist['losses'])
	plt.plot(hist['iters'], all_losses, alpha=0.5)
	plt.twinx().plot(hist['iters'], hist['acc'], alpha=1.0, color='black')
	# plt.twinx().plot(hist['iters'], hist['ent'], alpha=1.0, color='red')
	plt.show()


if __name__=='__main__':
	main()
