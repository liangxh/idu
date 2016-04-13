#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.16
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle
from optparse import OptionParser

import datica
import validatica
from const import N_EMO, DIR_EXT_MODEL, DIR_EXT_TEST
from inputadapter import InputAdapter
from lstmext import LstmClassifier

class lstmextscript:
	def __init__(self):
		self.init_default_options()
		self.add_extra_options()
		
	def init_default_options(self):
		parser = OptionParser()

		# necessary
		parser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
		
		# optional
		parser.add_option('-u', '--unigram', action='store_true', dest='unigram', default = False)
		parser.add_option('-r', '--resume', action='store_true', dest='resume', default = False)

		# debug
		parser.add_option('-y', '--ydim', action='store', type='int', dest='ydim', default = N_EMO)
		parser.add_option('-n', '--n_samples', action='store', dest='n_samples', default = None)

		# especially for gpu
		parser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)

		self.optparser = parser

	def add_extra_options(self):
		'''
		more options can be added for init_embedder, and so on
		'''
		pass

	def init_folder(self):
		'''
		mkdir if the necessary folders do not exist
		'''

		for dname = [DIR_EXTMODEL, DIR_EXTTEST]
			if not os.path.isdir(dname):
				os.mkdir(dname)

	def init_embedder(self, dataset, fname_embedder):
		'''
		initialize the embedder by load it from file if available
		or build the model by the dataset and save it
		'''

		#if os.path.exists(fname):
		#	print >> sys.stderr, 'embedding model %s found and loaded'%(fname)
		#	return RandEmbedder.load(fname)
		#else:
		#	def x_iterator(dataset):
		#		for set_x, set_y in dataset:
		#			for x in set_x:
		#				yield x
		#
		#	embedder = RandEmbedder()
		#	embedder.build(x_iterator(dataset), dim_proj)
		#	embedder.dump(fname)
		#	return embedder

		raise NotImplementedError

	def prepare_input(self, dataset, embedder):
		'''
		turn sequences of string into list of vectors
		'''
		
		def index_set(set_x_y):
			x, y = set_x_y
			new_x = [embedder.index(xi) for xi in x]
			print len(new_x)
			return (new_x, y)

		train, test, valid = dataset
		new_dataset = (index_set(train), index_set(test), index_set(valid))

		return new_dataset, embedder.get_Wemb()

	def run(self):
		'''
		the function to launch the script
		'''

		################### Preparation of Variables #######################
		print >> sys.stderr, 'lstmextscript.run: [info] preparing variables ... ', 

		opts, args = self.optparser.parse_args() # initialized in init_default_options
		self.opts = opts                         # shared by self. for customized function

		n_emo = opts.ydim
		datalen = opts.n_samples
		dim_proj = opts.dim_proj

		prefix = opts.prefix
		fname_test = DIR_EXTTEST + '%s_test.pkl'%(prefix)
		fname_model = DIR_EXTMODEL + '%s_model.npz'%(prefix)
		fname_embedder = DIR_EXTMODEL + '%s_embedder.pkl'%(prefix)

		print >> sys.stderr, 'Done'

		#################### Preparation of Input ##############
		print >> sys.stderr, 'lstmextscript.run: [info] loading dataset ... ', 

		if opts.unigram:
			dataset = datica.load_unigram(n_emo, datalen) 
		else:
			dataset = datica.load_token(n_emo, datalen)

		def add_default_xsup(dataset):
			train, valid, test = dataset

			def add_xsup(set_x_y):
				x, y = set_x_y
				n_samples = len(x)
				xsup = [[0, ] for i in range(n_samples)]
				return (x, y, xsup)

			return add_xsup(train), add_xsup(valid), add_xsup(test)

		dataset = add_default_xsup(dataset)

		print >> sys.stderr, 'Done'

		print >> sys.stderr, 'lstmextscript.run: [info] initialization of embedder'
		embedder = self.init_embedder(dataset, fname_embedder)

		print >> sys.stderr, 'lstmextscript.run: [info] preparing input'
		dataset, Wemb = self.prepare_input(dataset, embedder)

		#################### Preparation for Output ############
		self.init_folder()
	
		#################### Training ##########################
		print >> sys.stderr, 'lstmextscript.run: [info] start training'
	
		classifier = LstmClassifier()

		if not opts.resume:
			res = classifier.train(
				dataset = dataset,
				Wemb = Wemb,
				ydim = n_emo,

				fname_model = fname_model,

				batch_size = opts.batch_size,
				valid_batch_size = opts.batch_size,
			)
		elif not os.path.exists(fname_model):
			print >> sys.stderr, 'model %s not found'%(fname_model)
			return
		else:
			res = classifier.train(
				dataset = dataset,
				Wemb = Wemb,
				ydim = n_emo,

				fname_model = fname_model,
				reload_model = True,

				batch_size = opts.batch_size,
				valid_batch_size = opts.batch_size,
			)
		#else:
		#	print >> sys.stderr, 'lstm model %s found and loaded'%(fname_model)
		#	classifier.load(fname_model)

		###################### Test ##############################
		test_x, test_y, test_xsup = dataset[2]
		preds_prob = classifier.classify(test_x, test_xsup)
		cPickle.dump((test_y, preds_prob), open(fname_test, 'w'))

		###################### Report ############################
		validatica.report(test_y, preds_prob, DIR_TEST + prefix)

if __name__ == '__main__':
	main()

