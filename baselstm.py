#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.23
Description: Interface for Lstm
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
import cPickle

import theano
import numpy as np

from utils import baselstmtool as lstmtool
from utils.logger import Logger

from optparse import OptionParser

logger = Logger()

FNAME_MODEL = 'data/lstm_model.npz'
FNAME_TEST = None

class LstmClassifier:
	def __init__(self):
		pass

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
		x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)

		for idx, s in enumerate(seqs):
			x[:lengths[idx], idx] = s
			x_mask[:lengths[idx], idx] = 1.
		
		return x, x_mask

	@classmethod
	def prepare_data(self, seqs, labels, maxlen = None):
		x, x_mask = self.prepare_x(seqs)
		return x, x_mask, labels
	
	def train(self,
		dataset, Wemb,

		# model params		
		ydim, n_words,
		dim_proj = 128,
		encoder = 'lstm',
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

		model_options = locals().copy()
		cPickle.dump(model_options, open('%s.pkl'%(fname_model), 'wb'), -1) # why -1??

		params = lstmtool.init_params(model_options, Wemb)

		if reload_model:
			if os.path.exists(fname_model):
				lstmtool.load_params(fname_model, params)
			else:
				logger.warning('model %s not found'%(fname_model))
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
		
		grads = theano.tensor.grad(cost, wrt = tparams.values())
		f_grad = theano.function([x, mask, y], grads, name = 'f_grad')

		lr = theano.tensor.scalar(name = 'lr')
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

		return train_err, valid_err, test_err, end_time - start_time

from tfcoder import TfCoder

def randWemb(n_words, dim_proj):
	randn = np.random.rand(n_words, dim_proj)
	return (0.01 * randn).astype(theano.config.floatX)

def main():
	import cPickle
	import tfcoder	
	from const import PKL_TFCODER, N_EMO

	optparser = OptionParser()
	optparser.add_option('-y', '--ydim', action='store', type='int', dest='ydim') #, default=N_EMO
	optparser.add_option('-n', '--n_samples', action='store', type='int', dest='n_samples', default = None)
	optparser.add_option('-d', '--dim_proj', action='store', type='int', dest='dim_proj') #, default = 128
	optparser.add_option('-p', '--prefix', action='store', type='str', dest='prefix') #, default = 128

	optparser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)
	optparser.add_option('-v', '--valid_freq', action='store', type='int', dest='valid_freq', default = 10000)
	opts, args = optparser.parse_args()

	coder = cPickle.load(open(PKL_TFCODER, 'r'))
	n_emo = opts.ydim
	datalen = opts.n_samples
	dim_proj = opts.dim_proj

	fname_model = 'output/%s_model.npz'%(opts.prefix)
	fname_result = 'output/%s_test.pkl'%(opts.prefix)
	fname_valid_prefix = 'output/%s'%(opts.prefix)

	import baseunidatica as unidatica
	dataset = unidatica.load(n_emo, datalen)

	Wemb = randWemb(coder.n_code(), dim_proj)

	lstm = LstmClassifier()
	res = lstm.train(
			dataset = dataset,
			Wemb = Wemb,

			ydim = n_emo,
			dim_proj = dim_proj,
			n_words = coder.n_code(),
			fname_model = fname_model,
			reload_model = False,

			batch_size = opts.batch_size,
			valid_batch_size = opts.batch_size,
			validFreq = opts.valid_freq,
		)

	test_x, test_y = dataset[2]
	preds_prob = lstm.classify(test_x)
	cPickle.dump((test_y, preds_prob), open(fname_result, 'w'))

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

	optparser.add_option('-d', '--dim_proj', action='store', type='int', dest=None) #, default = 128
	optparser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)
	optparser.add_option('-v', '--valid_freq', action='store', type='int', dest='valid_freq', default = 10000)
	opts, args = optparser.parse_args()

	prefix = opts.prefix
	n_emo = opts.ydim
	datalen = opts.n_samples

	fname_model = 'output/%s_model.npz'%(prefix)
	fname_result = 'output/%s_test.pkl'%(prefix + '_valid')
	fname_valid_prefix = 'output/%s'%(prefix + '_valid')

	valid(n_emo, datalen, fname_model, fname_result, fname_valid_prefix)

if __name__ == '__main__':
	main()
	#main_valid()

