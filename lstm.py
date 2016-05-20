#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.23
Description: Interface for Lstm
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
import cPickle

import theano
import numpy as np

from utils import lstmtool
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
		self.tparams = tparams

		return train_err, valid_err, test_err, end_time - start_time

'''
from tfcoder import TfCoder

def randWemb(n_words, dim_proj):
	randn = np.random.rand(n_words, dim_proj)
	return (0.01 * randn).astype(theano.config.floatX)

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

def test():	
	ydim = 2
	xdim = 3
	n_word = 20
	n_length = 5
	
	Wemb = np.random.random((n_word, xdim))

	def random_xy(n_samples):
		x = [[np.random.randint(n_word) for j in range(n_length)] for i in range(n_samples)]
		y = [np.random.randint(ydim) for j in range(n_samples)]

		return (x, y)
	
	train = random_xy(2000)
	valid = random_xy(200)
	test = random_xy(50)
	dataset = (train, valid, test)

	clf = LstmClassifier()
	res = clf.train(
			dataset = dataset,
			Wemb = Wemb,
			ydim = ydim,

			validFreq = 100,
			saveFreq = 100,
		
			fname_model = 'output/lstmlr_model.npz',
			max_epochs = 2,
		)

	clf2 = LstmClassifier()
	clf2.load(fname_model)
	print clf2.pred_error(train)
	print clf2.pred_error(valid)
	print clf2.pred_error(test)

def main():
	import datica
	import validatica
	
	import wemb_rand
	from wordembedder import WordEmbedder
	from const import N_EMO, DIR_MODEL, DIR_TEST
	
	def init_embedder(dataset, fname_embedder, xdim = None):
		'''
		initialize the embedder by load it from file if available
		or build the model by the dataset and save it
		'''

		if os.path.exists(fname_embedder):
			print >> sys.stderr, 'main: [info] embedding model %s found and loaded'%(fname_embedder)
			return WordEmbedder.load(fname_embedder)
		else:
			assert xdim is not None

			def x_iterator(dataset):
				train, valid, test = dataset
				for x, y in [train, valid]:
					for xi in x:
						yield xi

			embedder = WordEmbedder(*wemb_rand.build(x_iterator(dataset), xdim))
			embedder.dump(fname_embedder)
		
			return embedder

	def prepare_input(dataset, embedder):
		'''
		turn sequences of string into list of vectors
		'''
		def index_xy(xy):
			x, y = xy
			new_x = [embedder.index(xi) for xi in x]
			return (new_x, y)

		train, test, valid = dataset
		new_dataset = (index_xy(train), index_xy(test), index_xy(valid))

		return new_dataset, embedder.get_Wemb()


	# Initialization of OptionParser
	optparser = OptionParser()

	optparser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
	optparser.add_option('-o', '--dir_output', action='store', type = 'str', dest='dir_output', default = 'data/dataset/')
	optparser.add_option('-x', '--dir_x', action='store', type = 'str', dest='dir_x', default = 'data/dataset/unigram/')

	optparser.add_option('-d', '--dim_proj', action='store', type = 'int', dest='dim_proj')
	optparser.add_option('-y', '--ydim', action='store', type='int', dest='ydim', default = N_EMO)
	optparser.add_option('-n', '--n_samples', action='store', dest='n_samples', default = None)

	optparser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)
	opts, args = optparser.parse_args()

	prefix = opts.prefix
	
	# Prepare filenames
	dir_test = opts.dir_output + 'test/'
	dir_model = opts.dir_output + 'model/'

	fname_test = dir_test + '%s_test.pkl'%(prefix)
	fname_model = dir_model + '%s_model.npz'%(prefix)
	fname_embedder = dir_model + '%s_embedder.pkl'%(prefix)

	dataset = datica.load_data(opts.dir_x, opts.ydim, opts.n_samples)

	print >> sys.stderr, 'main: [info] initialization of embedder'
	embedder = init_embedder(dataset, fname_embedder, opts.dim_proj)

	print >> sys.stderr, 'main: [info] preparing input'
	dataset, Wemb = prepare_input(dataset, embedder)

	print >> sys.stderr, 'lstmextscript.run: [info] start training'
	clf = LstmClassifier()

	res = clf.train(
			dataset = dataset,
			Wemb = Wemb,
			ydim = opts.ydim,
			fname_model = fname_model,
			batch_size = opts.batch_size,

			max_epochs = 5000,
		)

	test_x, test_y = dataset[2]
	proba = clf.predict_proba(test_x)
	cPickle.dump((test_y, proba), open(fname_test, 'w'))

	###################### Report ############################
	validatica.report(test_y, proba, DIR_TEST + prefix)

if __name__ == '__main__':
	main()
