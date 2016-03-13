#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.23
Description: LstmClassifier is a class re-constructed from DeepLearningTutorial.code.lstm.py
	whose interface supports self-customized vector as input
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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict

from utils import lstmtool
from utils.logger import Logger

logger = Logger()

FNAME_MODEL = 'data/new_lstm_model.npz'

# initization for randomization
RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)

def seqs_to2D(x):
	return [[[xi, xi] for xi in seq] for seq in x]

def np_floatX(data):
	'''
	easy creation np.array of theano.config.floatX 
	'''	
	return np.asarray(data, dtype = theano.config.floatX)

def _p(pp, name):
	'''
	just for naming
	'''	
	return '%s_%s' % (pp, name)

def ortho_weight(ndim):
	'''
	create a random matrix(ndim x ndim) 
	'''
	W = np.random.randn(ndim, ndim)
	u, s, v = np.linalg.svd(W)
	return u.astype(theano.config.floatX)

def adadelta(lr, tparams, grads, x, mask, y, cost):
	'''
	same as in lstmtool.py
	'''
	zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),name='%s_grad' % k)
				for k, p in tparams.iteritems()]

	running_up2 = [theano.shared(p.get_value() * np_floatX(0.), name='%s_rup2' % k)
				for k, p in tparams.iteritems()]

	running_grads2 = [theano.shared(p.get_value() * np_floatX(0.), name='%s_rgrad2' % k)
				for k, p in tparams.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
			 for rg2, g in zip(running_grads2, grads)]
	f_grad_shared = theano.function(
				[x, mask, y], cost,
				updates=zgup + rg2up,
				name='adadelta_f_grad_shared'
		)
	
	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
			 for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]

	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
			 for ru2, ud in zip(running_up2, updir)]

	param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
	
	f_update = theano.function(
			[lr], [],
			updates = ru2up + param_up,
			on_unused_input = 'ignore',
			name = 'adadelta_f_update'
		)
	
	return f_grad_shared, f_update

class LstmClassifier:
	def __init__(self):
		pass

	########################### Building Model ############################################
	def init_params(self, options):
		'''
		Init the LSTM parameter:
		'''
		params = OrderedDict()
		# params['Wemb'] removed 

		# params for LSTM
		prefix = 'lstm'
		N = 4 # magic number?
		params[_p(prefix, 'W')] = np.concatenate([ortho_weight(options['dim_proj']) for i in range(N)], axis=1)
		params[_p(prefix, 'U')] = np.concatenate([ortho_weight(options['dim_proj']) for i in range(N)], axis=1)
		params[_p(prefix, 'b')] = np.zeros((N * options['dim_proj'],)).astype(theano.config.floatX)

		# params for classifier
		params['U'] = 0.01 * np.random.randn(options['dim_proj'], options['ydim']).astype(theano.config.floatX)
		params['b'] = np.zeros((options['ydim'],)).astype(theano.config.floatX)

		return params

	def init_layer(self, tparams, state_below, options, mask=None):
		'''
		copied from lstmtool
		'''
		prefix = 'lstm'

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
			preact = T.dot(h_, tparams[_p(prefix, 'U')])
			preact += x_

			i = T.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
			f = T.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
			o = T.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
			c = T.tanh(_slice(preact, 3, options['dim_proj']))

			c = f * c_ + i * c
			c = m_[:, None] * c + (1. - m_)[:, None] * c_

			h = o * T.tanh(c)
			h = m_[:, None] * h + (1. - m_)[:, None] * h_

			return h, c

		state_below = (
			T.dot(state_below, tparams[_p(prefix, 'W')]) +
			tparams[_p(prefix, 'b')]
			)

		dim_proj = options['dim_proj']
		rval, updates = theano.scan(
					_step,
					sequences = [mask, state_below],
					outputs_info = [
						T.alloc(np_floatX(0.), n_samples, dim_proj),
						T.alloc(np_floatX(0.), n_samples, dim_proj)
						],
					name = _p(prefix, '_layers'),
					n_steps = nsteps
				)
		return rval[0]

	def build_model(self, tparams, options):
		trng = RandomStreams(RANDOM_SEED)

		# Used for dropout.
		use_noise = theano.shared(np_floatX(0.))

		x = T.tensor3('x', dtype = theano.config.floatX)
		mask = T.matrix('mask', dtype = theano.config.floatX)
		y = T.vector('y', dtype='int64')

		n_timesteps = x.shape[0]
		n_samples = x.shape[1]

		proj = self.init_layer(tparams, x, options, mask = mask)

		# specific for LSTM
		proj = (proj * mask[:, :, None]).sum(axis=0)
		proj = proj / mask.sum(axis=0)[:, None]

		if options['use_dropout']:
			proj = lstmtool.dropout_layer(proj, use_noise, trng)

		pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])

		f_pred_prob = theano.function([x, mask], pred, name = 'f_pred_prob')
		f_pred = theano.function([x, mask], pred.argmax(axis = 1), name = 'f_pred')

		off = 1e-8
		if pred.dtype == 'float16':
			off = 1e-6

		cost = - T.log(pred[T.arange(n_samples), y] + off).mean()

		return use_noise, x, mask, y, f_pred_prob, f_pred, cost

	########################## Classification ###################################
	def load(self, 
		#ydim,
		#dim_proj = 128,
		#use_dropout = True,
		fname_model = FNAME_MODEL, 
	):
		## not changed yet to self.init_params or whatever
		model_options = locals().copy()

		train_params = cPickle.load(open('%s.pkl'%(fname_model), 'r')) # why -1??
		model_options.update(train_params)

		params = self.init_params(model_options)

		self.load_params(fname_model, params)
		tparams = lstmtool.init_tparams(params)

		use_noise, x, mask, y, f_pred_prob, f_pred, cost = self.build_model(tparams, model_options)

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

		xdim = len(seqs[0][0])

		n_samples = len(seqs)
		maxlen = np.max(lengths)

		x = np.zeros((maxlen, n_samples, xdim)).astype('int64')
		x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)

		for idx, s in enumerate(seqs):
			x[:lengths[idx], idx, :] = s
			x_mask[:lengths[idx], idx] = 1.
		
		return x, x_mask

	@classmethod
	def prepare_data(self, seqs, labels, maxlen = None):
		x, x_mask = self.prepare_x(seqs)
		return x, x_mask, labels

	@classmethod
	def x_1Dto2D(self, dataset):
		new_dataset = []

		train_x, train_y = dataset[0]
		new_train = (seqs_to2D(train_x), train_y)		

		valid_x, valid_y = dataset[1]
		new_valid = (seqs_to2D(valid_x), valid_y)

		test_x, test_y = dataset[2]
		new_test = (seqs_to2D(test_x), test_y)

		return new_train, new_valid, new_test
	
	def train(self,
		dataset,

		# model params		
		ydim,
		use_dropout = True,
		reload_model = False,
		fname_model = FNAME_MODEL,
		
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

		model_options = locals().copy()

		if isinstance(train[0][0][0], list):
			model_options['dim_proj'] = len(train[0][0][0])
		else:
			'''
			as i don't know what's wrong with init_layer when dim_proj = 1
			so i map x -> (x, x) to change dim_proj as 2
			'''
			train, valid, test = LstmClassifier.x_1Dto2D(dataset)
			model_options['dim_proj'] = 2

		params = self.init_params(model_options)

		if reload_model:
			if os.path.exists(fname_model):
				lstmtool.load_params(fname_model, params)
			else:
				logger.warning('model %s not found'%(fname_model))
				return None
		
		tparams = lstmtool.init_tparams(params)
		use_noise, x, mask, y, f_pred_prob, f_pred, cost = self.build_model(tparams, model_options)

		# preparing functions for training
		logger.info('preparing functions')

		if decay_c > 0.:
			decay_c = theano.shared(np_floatX(decay_c), name='decay_c')
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

		uidx = 0	   # number of update done
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
						cPickle.dump(model_options, open('%s.pkl'%(fname_model), 'wb'), -1) # why -1??

					if np.mod(uidx, validFreq) == 0:
						'''
						check prediction error at %validFreq
						'''
						use_noise.set_value(0.)
						
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

		return train_err, valid_err, test_err, end_time - start_time

from tfcoder import TfCoder

def main():
	from const import N_EMO

	n_emo = 2 #N_EMO # 2

	import unidatica
	dataset = unidatica.load(n_emo, 1000) #, 1000

	lstm = LstmClassifier()
	res = lstm.train(
			dataset = dataset,
			ydim = n_emo,
			fname_model = FNAME_MODEL,
			reload_model = True,
		)

def valid(outfile = 'output/new_lstm_result.pkl'):
	import unidatica
	from const import N_EMO

	n_emo = 2 #N_EMO
	dataset = unidatica.load(n_emo, 1000)
	
	lstm = LstmClassifier()
	lstm.load()

	test_x, test_y = dataset[2]
	if len(test_x[0][0]) == 1:
		test_x = seqs_to2D(test_x)

	preds_prob = lstm.classify(test_x)
	cPickle.dump((test_y, preds_prob), open(outfile, 'w'))

if __name__ == '__main__':
	main()
	#valid()

