#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.19
Description: Interface for Lstm-LR
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
import cPickle

import numpy as np
import theano
import theano.tensor as T
import theano.config.floatX  as floatX
from collections import OrderedDict

from utils import lstmtool
from utils.logger import Logger

from optparse import OptionParser

logger = Logger()

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Set the random number generators' seeds for consistency
SEED = 123 
numpy.random.seed(SEED)

###################### Shortcut ######################################
def numpy_floatX(data):
	return np.asarray(data, dtype = floatX)

def _p(pp, name):
	return '%s_%s' % (pp, name)

###################### Zip and Unzip #################################
def zipp(params, tparams):
	'''
	When we reload the model. Needed for the GPU stuff.
	'''
	for kk, vv in params.iteritems():
		tparams[kk].set_value(vv)

def unzip(zipped):
	'''
	When we pickle the model. Needed for the GPU stuff.
	'''
	new_params = OrderedDict()
	for kk, vv in zipped.iteritems():
		new_params[kk] = vv.get_value()
	return new_params

##################### Tools ##################################################
def get_minibatches_idx(n, minibatch_size, shuffle=False):
	'''
	get batches of idx for range(1, n) and shuffle if needed
	'''
	idx_list = numpy.arange(n, dtype="int32")

	if shuffle:
		numpy.random.shuffle(idx_list)

	minibatches = []
	minibatch_start = 0
	for i in range(n // minibatch_size):
		minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
		minibatch_start += minibatch_size

	if (minibatch_start != n):
		# Make a minibatch out of what is left
		minibatches.append(idx_list[minibatch_start:])

	return zip(range(len(minibatches)), minibatches)

def load_params(path, params):
	pp = numpy.load(path)
	for kk, vv in params.iteritems():
		if kk not in pp:
			raise Warning('%s is not in the archive' % kk)
		params[kk] = pp[kk]

	return params

def init_tparams(params):
	tparams = OrderedDict()
	for kk, pp in params.iteritems():
		tparams[kk] = theano.shared(params[kk], name=kk)
	return tparams

def ortho_weight(ndim):
	W = np.random.randn(ndim, ndim)
	u, s, v = np.linalg.svd(W)
	return u.astype(floatX)

def dropout_layer(state_before, use_noise, trng):
	'''
	return a dropout layer
	$state_before refers to x, rename it later maybe
	'''
	proj = tensor.switch(
			use_noise,
			(state_before
				* trng.binomial(state_before.shape, p = 0.5, n = 1, dtype = state_before.dtype)),
			state_before * 0.5
		)
	return proj

def param_init_lstm(options, params, prefix='lstm'):
	N = 4 # input~/input/output/forget gate

	W = np.concatenate([ortho_weight(options['dim_proj']) for i in range(N)], axis=1)
	params[_p(prefix, 'W')] = W

	U = np.concatenate([ortho_weight(options['dim_proj']) for i in range(N)], axis=1)
	params[_p(prefix, 'U')] = U

	b = np.zeros((N * options['dim_proj'],))
	params[_p(prefix, 'b')] = b.astype(floatX)

	return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
	nsteps = state_below.shape[0]
	if state_below.ndim == 3:
		n_samples = state_below.shape[1]
	else:
		n_samples = 1

	assert mask is not None

	def _slice(_x, n, dim):
		if _x.ndim == 3:
			return _x[:, :, n * dim:(n + 1) * dim]
		return _x[:, n * dim:(n + 1) * dim]

	def _step(m_, x_, h_, c_):
		preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
		preact += x_

		i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
		f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
		o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
		c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

		c = f * c_ + i * c
		c = m_[:, None] * c + (1. - m_)[:, None] * c_

		h = o * tensor.tanh(c)
		h = m_[:, None] * h + (1. - m_)[:, None] * h_

		return h, c

	state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
				   tparams[_p(prefix, 'b')])

	dim_proj = options['dim_proj']
	rval, updates = theano.scan(
				_step,
				sequences = [mask, state_below],
				outputs_info = [
					T.alloc(np_floatX(0.), n_samples, dim_proj),
					T.alloc(np_floatX(0.), n_samples, dim_proj)
					],
				name=_p(prefix, '_layers'),
				n_steps = nsteps
			)
	return rval[0]

class LstmClassifier:
	def __init__(self):
		pass

	########################## Building Model ###################################
	def init_params(options, Wemb = None):
		'''
		initizalize params for every layer
		'''
		params = OrderedDict()

		# Embedding and LSTM
	   	params['Wemb'] = Wemb
		params = get_layer(options['encoder'])[0](options, params, prefix = options['encoder'])
		
		# logistic Regression
		params['U'] = 0.01 * numpy.random.randn(options['dim_proj'], options['ydim']).astype(floatX)
		params['b'] = numpy.zeros((options['ydim'],)).astype(floatX)

		return params

	def build_model(tparams, options):
		trng = RandomStreams(SEED)

		# Used for dropout.
		# a shared variable, changed in the training process to control whether to use noise or not
		use_noise = theano.shared(numpy_floatX(0.))

		# a matrix, whose shape is (n_timestep, n_samples) for theano.scan in training process
		x = T.matrix('x', dtype='int64')
		
		# a matrix, used to distinguish the valid elements in x
		mask = T.matrix('mask', dtype=floatX)

		# a vector of targets for $n_samples samples   
		y = T.vector('y', dtype='int64')

		n_timesteps = x.shape[0]
		n_samples = x.shape[1]

		# transfer x, the matrix of tids, into Wemb, the 'matrix' of embedding vectors 
		emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
							n_samples,
							options['dim_proj']])

		# the result of LSTM, a matrix of shape (n_timestep, n_samples, dim_proj)
		proj = get_layer(options['encoder'])[1](tparams, emb, options,
							prefix=options['encoder'],
							mask=mask)

		# mean pooling, a matrix of shape (n_samples, dim_proj)
		if options['encoder'] == 'lstm':
			proj = (proj * mask[:, :, None]).sum(axis=0)
			proj = proj / mask.sum(axis=0)[:, None]

		# add a dropout layer after mean pooling
		if options['use_dropout']:
			proj = dropout_layer(proj, use_noise, trng)

		pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])

		f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
		f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

		off = 1e-8
		if pred.dtype == 'float16':
		off = 1e-6

		cost = -T.log(pred[T.arange(n_samples), y] + off).mean()

		return use_noise, x, mask, y, f_pred_prob, f_pred, cost


	########################## Classification ###################################
	def load(self, 
		fname_model,
		encoder = 'lstm',
	
		#ydim, n_words,
		#dim_proj = 128,
		#use_dropout = True,	
	):

		model_options = locals().copy()

		train_params = cPickle.load(open('%s.pkl'%(fname_model), 'r')) # why -1??
		model_options.update(train_params)

		params = lstmtool.init_params(model_options, None)
		lstmtool.load_params(fname_model, params)
		tparams = lstmtool.init_tparams(params)

		use_noise, x, mask, y, f_pred_prob, f_pred, cost = lstmtool.build_model(tparams, model_options)

		self.f_pred = f_pred
		self.f_pred_prob = f_pred_prob

	def classify_batch(self, seqs):
		x, x_mask = self.prepare_x(seqs)
		#ps = self.f_pred(x, x_mask)
		pds = self.f_pred_prob(x, x_mask)

		return pds

	def classify(self, seqs, batch_size = 64):
		if not isinstance(seqs[0], list):
			seqs = [seqs, ]
			pred_probs = self.classify_batch(seqs)

			logger.warning('not examined yet, please check')
			return pred_probs[0]
		else:
			kf = lstmtool.get_minibatches_idx(len(seqs), batch_size)
		
			#preds = []
			pred_probs = []

			for _, idx in kf:
				pds = self.classify_batch([seqs[i] for i in idx])	
				#preds.extend(ps)
				pred_probs.extend(pds)

			return pred_probs

	######################## Training ##########################################

	@classmethod
	def prepare_x(self, seqs):
		'''
		create two 2D-Arrays (seqs and mask)
		'''
		lengths = [len(s) for s in seqs]

		n_samples = len(seqs)
		maxlen = np.max(lengths)

		x = np.zeros((maxlen, n_samples)).astype('int64')
		x_mask = np.zeros((maxlen, n_samples)).astype(theano.floatX)

		for idx, s in enumerate(seqs):
			x[:lengths[idx], idx] = s
			x_mask[:lengths[idx], idx] = 1.
		
		return x, x_mask

	@classmethod
	def prepare_data(self, seqs, labels, maxlen = None):
		x, x_mask = self.prepare_x(seqs)
		return x, x_mask, labels
	
	def train(self,
		dataset, Wemb, ydim,
		
		# model params		
		use_dropout = True,
		reload_model = False,
		fname_model = None,
		
		# training params
		validFreq = 1000,
		saveFreq = 1000,
		patience = 10,
		max_epochs = 5000,
		decay_c = 0.,
		lrate = 0.0001,
		batch_size = 16,
		valid_batch_size = 64,
		optimizer = lstmtool.adadelta,
		noise_std = 0., 

		# debug params
		dispFreq = 10,
	):
		train, valid, test = dataset

		# building model
		logger.info('building model...')

		dim_proj = Wemb.shape[1] # numpy.ndarray expected

		model_options = locals().copy()
		model_options['dim_proj'] = dim_proj
		model_options['encoder'] = 'lstm'

		model_config = {
			'ydim':ydim,
			'dim_proj':dim_proj,
			'use_dropout':use_dropout,
			'fname_model':fname_model,
		}
		cPickle.dump(model_config, open('%s.pkl'%(fname_model), 'wb'), -1) # why -1??

		params = lstmtool.init_params(model_options, Wemb)

		if reload_model:
			if os.path.exists(fname_model):
				lstmtool.load_params(fname_model, params)
			else:
				logger.warning('model %s not found'%(fname_model))
				return None
		elif Wemb is None:
			logger.warning('Wemb is missing for training LSTM')
			return None
		
		tparams = lstmtool.init_tparams(params)
		use_noise, x, mask, y, f_pred_prob, f_pred, cost = lstmtool.build_model(tparams, model_options)

		# preparing functions for training
		logger.info('preparing functions')

		if decay_c > 0.:
			decay_c = theano.shared(lstmtool.numpy_floatX(decay_c), name='decay_c')
			weight_decay = 0.
			weight_decay += (tparams['U'] ** 2).sum()
			weight_decay *= decay_c
			cost += weight_decay
	
		f_cost = theano.function([x, mask, y], cost, name = 'f_cost')
		
		grads = T.grad(cost, wrt = tparams.values())
		f_grad = theano.function([x, mask, y], grads, name = 'f_grad')

		lr = T.scalar(name = 'lr')
		f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)

		kf_valid = lstmtool.get_minibatches_idx(len(valid[0]), valid_batch_size)
		kf_test = lstmtool.get_minibatches_idx(len(test[0]), valid_batch_size)

		if validFreq == None:
			validFreq = len(train[0]) / batch_size
		
		if saveFreq == None:
			saveFreq = len(train[0]) / batch_size
		
		history_errs = []
		best_p = None
		bad_count = 0

		uidx = 0       # number of update done
		estop = False  # early stop

		# training
		logger.info('start training...')

		start_time = time.time()

		try:
			for eidx in xrange(max_epochs):
				n_samples = 0
				
				kf = lstmtool.get_minibatches_idx(len(train[0]), batch_size, shuffle = True)
				
				for _, train_index in kf:
					uidx += 1
					use_noise.set_value(1.)

					x = [train[0][t] for t in train_index]
					y = [train[1][t] for t in train_index]

					x, mask = self.prepare_x(x)
					n_samples += x.shape[1]

					cost = f_grad_shared(x, mask, y)
					f_update(lrate)
					
					if np.isnan(cost) or np.isinf(cost):
						'''
						NaN of Inf encountered
						'''
						logger.warning('NaN detected')
						return 1., 1., 1.
					
					if np.mod(uidx, dispFreq) == 0:
						'''
						display progress at $dispFreq
						'''
						logger.info('Epoch %d Update %d Cost %f'%(eidx, uidx, cost))

					if np.mod(uidx, saveFreq) == 0:
						'''
						save new model to file at $saveFreq
						'''
						logger.info('Model update')
						
						if best_p is not None:
							params = best_p
						else:
							params = lstmtool.unzip(tparams)
					
						np.savez(fname_model, history_errs = history_errs, **params)

					if np.mod(uidx, validFreq) == 0:
						'''
						check prediction error at %validFreq
						'''
						use_noise.set_value(0.)
						
						logger.info('Validation ....')

						# not necessary	
						train_err = lstmtool.pred_error(f_pred, self.prepare_data, train, kf)
						
						valid_err = lstmtool.pred_error(f_pred, self.prepare_data, valid, kf_valid)
						test_err = lstmtool.pred_error(f_pred, self.prepare_data, test, kf_test)

						history_errs.append([valid_err, test_err])
						if (uidx == 0 or valid_err <= np.array(history_errs)[:, 0].min()):
							best_p = lstmtool.unzip(tparams)
							bad_count = 0
						
						logger.info('prediction error: train %f valid %f test %f'%(
								train_err, valid_err, test_err)
							)
						if (len(history_errs) > patience and
							valid_err >= np.array(history_errs)[:-patience, 0].min()):
							bad_count += 1
							if bad_count > patience:
								logger.info('Early stop!')
								estop = True
								break

				logger.info('%d samples seen'%(n_samples))
				if estop:
					break
	
		except KeyboardInterrupt:
			print logger.debug('training interrupted by user')

		end_time = time.time()

		if best_p is not None:
			lstmtool.zipp(best_p, tparams)
		else:
			best_p = lstmtool.unzip(tparams)

		use_noise.set_value(0.)
		
		kf_train = lstmtool.get_minibatches_idx(len(train[0]), batch_size)
		train_err = lstmtool.pred_error(f_pred, self.prepare_data, train, kf_train)
		valid_err = lstmtool.pred_error(f_pred, self.prepare_data, valid, kf_valid)
		test_err = lstmtool.pred_error(f_pred, self.prepare_data, test, kf_test)
 
		logger.info('prediction error: train %f valid %f test %f'%(
				train_err, valid_err, test_err)
			)
		
		np.savez(
			fname_model,
			train_err = train_err,
			valid_err = valid_err,
			test_error = test_err,
			history_errs = history_errs, **best_p
			)

		logger.info('totally %d epoches in %.1f sec'%(eidx + 1, end_time - start_time))

		self.f_pred_prob = f_pred_prob
		self.f_pred = f_pred
		self.tparams = tparams

		return train_err, valid_err, test_err, end_time - start_time

'''
from tfcoder import TfCoder

def randWemb(n_words, dim_proj):
	randn = np.random.rand(n_words, dim_proj)
	return (0.01 * randn).astype(theano.floatX)

def main():
	import cPickle
	from const import N_EMO	

	########################### OptionParser ################################
	optparser = OptionParser()
	
	# necessary
	optparser.add_option('-d', '--dim_proj', action='store', type='int', dest='dim_proj') #, default = 128
	optparser.add_option('-p', '--prefix', action='store', type='str', dest='prefix') #, default = 128

	# debug
	optparser.add_option('-y', '--ydim', action='store', type='int', dest='ydim', default = N_EMO)
	optparser.add_option('-n', '--n_samples', action='store', type='int', dest='n_samples', default = None)

	# especially for gpu
	optparser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)
	optparser.add_option('-r', '--resume', action='store_true', dest='resume', default = False)

	opts, args = optparser.parse_args()

	########################## LSTM Training ####################################
	from const import PKL_TFCODER
	coder = cPickle.load(open(PKL_TFCODER, 'r'))
	n_emo = opts.ydim
	datalen = opts.n_samples
	dim_proj = opts.dim_proj

	fname_model = 'output/%s_model.npz'%(opts.prefix)
	fname_result = 'output/%s_test.pkl'%(opts.prefix)
	fname_valid_prefix = 'output/%s'%(opts.prefix)

	Wemb = randWemb(coder.n_code(), dim_proj)

	import baseunidatica as unidatica
	dataset = unidatica.load(n_emo, datalen)

	lstm = LstmClassifier()
	res = lstm.train(
			dataset = dataset,
			Wemb = Wemb,
			ydim = n_emo,

			fname_model = fname_model,
			reload_model = opts.resume,

			batch_size = opts.batch_size,
			valid_batch_size = opts.batch_size,
		)
	######################### Test #############################################
	test_x, test_y = dataset[2]
	preds_prob = lstm.classify(test_x)
	cPickle.dump((test_y, preds_prob), open(fname_result, 'w'))

	######################### Graph ############################################
	import validatica
	validatica.report(test_y, preds_prob, fname_valid_prefix)


def valid(n_emo, datalen, fname_model, fname_result, fname_valid_prefix):
	import cPickle
	import tfcoder	
	from const import PKL_TFCODER, N_EMO

	#coder = cPickle.load(open(PKL_TFCODER, 'r'))
	#n_emo = 2 #N_EMO

	lstm = LstmClassifier()
	lstm.load(
			#ydim = n_emo,
			#n_words = coder.n_code(),
			fname_model = fname_model,
		)

	import baseunidatica as unidatica
	dataset = unidatica.load(n_emo, datalen)
	test_x, test_y = dataset[2]

	preds_prob = lstm.classify(test_x)
	cPickle.dump((test_y, preds_prob), open(fname_result, 'w'))

	import validatica
	validatica.report(test_y, preds_prob, fname_valid_prefix)


def main_valid():
	optparser = OptionParser()
	optparser.add_option('-p', '--prefix', action='store', type='str', dest='prefix')
	optparser.add_option('-n', '--n_samples', action='store', type='int', dest='n_samples', default = None)
	optparser.add_option('-y', '--ydim', action='store', type='int', dest='ydim') #, default=N_EMO

	optparser.add_option('-d', '--dim_proj', action='store', type='int') #, default = 128
	opts, args = optparser.parse_args()

	prefix = opts.prefix
	n_emo = opts.ydim
	datalen = opts.n_samples

	fname_model = 'output/%s_model.npz'%(prefix)
	fname_result = 'output/%s_test.pkl'%(prefix + '_valid')
	fname_valid_prefix = 'output/%s'%(prefix + '_valid')

	valid(n_emo, datalen, fname_model, fname_result, fname_valid_prefix)
'''

if __name__ == '__main__':
	main()
	#main_valid()

