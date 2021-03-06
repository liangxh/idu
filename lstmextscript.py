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

import daticaext
import validatica

from lstmext import LstmClassifier

import wemb_rand
from wordembedder import WordEmbedder

N_EMO = 25
from const import DIR_EXTTEST, DIR_EXTMODEL

class LstmExtScript:
	def __init__(self):
		self.init_default_options()
		self.add_extra_options()
		
	def init_default_options(self):
		parser = OptionParser()

		# necessary
		parser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
		parser.add_option('-o', '--dir_output', action='store', type = 'str', dest='dir_output', default = None)
		parser.add_option('-x', '--dname_x', action='store', type = 'str', dest='dname_x')
		parser.add_option('-s', '--dname_xsup', action='store', type = 'str', dest='dname_xsup', default = None)		

		# optional
		parser.add_option('-r', '--resume', action='store_true', dest='resume', default = False)
		parser.add_option('-l', '--load', action='store_true', dest='load', default = False)	

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
		self.optparser.add_option('-d', '--dim_proj', action='store', type = 'int', dest='dim_proj') # , default = 128

	def init_folder(self, dnames):
		'''
		mkdir if the necessary folders do not exist
		'''

		for dname in dnames:
			if not os.path.isdir(dname):
				os.mkdir(dname)

	def init_embedder(self, dataset, fname_embedder):
		'''
		initialize the embedder by load it from file if available
		or build the model by the dataset and save it
		'''

		if os.path.exists(fname_embedder):
			print >> sys.stderr, 'embedding model %s found and loaded'%(fname_embedder)
			return WordEmbedder.load(fname_embedder)
		else:
			def x_iterator(dataset):
				for set_x, set_y, set_xsup in dataset:
					for x in set_x:
						yield x

			embedder = WordEmbedder(*wemb_rand.build(x_iterator(dataset), self.opts.dim_proj))
			embedder.dump(fname_embedder)
		
			return embedder

	def prepare_input(self, dataset, embedder):
		'''
		turn sequences of string into list of vectors
		'''
		
		def index_set(set_x_y_xsup):
			x, y, xsup = set_x_y_xsup
			new_x = [embedder.index(xi) for xi in x]
			print len(new_x)
			return (new_x, y, xsup)

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
	
		if opts.dir_output is None:
			dir_test = DIR_EXTTEST
			dir_model = DIR_EXTMODEL
		else:
			dir_test = opts.dir_output + 'test/'
			dir_model = opts.dir_output + 'model/'

		fname_test = dir_test + '%s_test.pkl'%(prefix)
		fname_model = dir_model + '%s_model.npz'%(prefix)
		fname_embedder = dir_model + '%s_embedder.pkl'%(prefix)

		print >> sys.stderr, 'Done'

		#################### Preparation of Input ##############
		print >> sys.stderr, 'lstmextscript.run: [info] loading dataset ... ', 

		dataset = daticaext.load_data(n_emo, opts.dname_x, opts.dname_xsup, datalen) 
		
		print >> sys.stderr, 'Done'

		#################### Preparation for Output ############
		self.init_folder([dir_test, dir_model])

		print >> sys.stderr, 'lstmextscript.run: [info] initialization of embedder'
		embedder = self.init_embedder(dataset, fname_embedder)

		print >> sys.stderr, 'lstmextscript.run: [info] preparing input'
		dataset, Wemb = self.prepare_input(dataset, embedder)
	
		#################### Training ##########################
		print >> sys.stderr, 'lstmextscript.run: [info] start training'
	
		classifier = LstmClassifier()

		if opts.load:
			classifier.load(fname_model)
		elif not opts.resume:
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

		###################### Test ##############################
		test_x, test_y, test_xsup = dataset[2]
		preds_prob = classifier.classify(test_x, test_xsup)
		cPickle.dump((test_y, preds_prob), open(fname_test, 'w'))

		###################### Report ############################
		validatica.report(test_y, preds_prob, dir_test + prefix)

def main():
	script = LstmExtScript()
	script.run()

if __name__ == '__main__':
	main()

