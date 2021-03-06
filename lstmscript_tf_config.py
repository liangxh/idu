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
from const import N_EMO, DIR_MODEL, DIR_TEST, DIR_DATA, DIR_UNIGRAM
from lstm import LstmClassifier

import wemb_randtf
from wordembedder import WordEmbedder

class LstmScriptRand:
	def __init__(self):
		self.init_default_options()
		self.add_extra_options()
		
	def init_default_options(self):
		parser = OptionParser()

		# necessary
		parser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
		parser.add_option('-c', '--fname_config', action='store', type = 'str', dest='fname_config')

		# optional
		parser.add_option('-r', '--resume', action='store_true', dest='resume', default = False)

		# debug
		parser.add_option('-n', '--n_samples', action='store', dest='n_samples', default = None)

		# especially for gpu
		parser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)

		self.optparser = parser

	def add_extra_options(self):
		self.optparser.add_option('-d', '--dim_proj', action='store', type = 'int', dest='dim_proj')
		self.optparser.add_option('-k', '--keep_rate', action='store', dest='keep_rate', type='float', default = 0.2)

	def init_folder(self):
		'''
		mkdir if the necessary folders do not exist
		'''
		if not os.path.isdir(DIR_MODEL):
			os.mkdir(DIR_MODEL)

		if not os.path.isdir(DIR_TEST):
			os.mkdir(DIR_TEST)

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
				for set_x, set_y in dataset:
					for x in set_x:
						yield x

			embedder = WordEmbedder(*wemb_randtf.build(x_iterator(dataset), self.opts.dim_proj, self.opts.keep_rate))
			embedder.dump(fname_embedder)
		
			return embedder

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
		print >> sys.stderr, 'lstmscript.run: [info] preparing variables ... ', 

		opts, args = self.optparser.parse_args() # initialized in init_default_options
		self.opts = opts                         # shared by self. for customized function
		
		datalen = opts.n_samples
		dim_proj = opts.dim_proj

		prefix = opts.prefix
		fname_test = DIR_TEST + '%s_test.pkl'%(prefix)
		fname_model = DIR_MODEL + '%s_model.npz'%(prefix)
		fname_embedder = DIR_MODEL + '%s_embedder.pkl'%(prefix)

		print >> sys.stderr, 'Done'


		#################### Preparation of Input ##############
		print >> sys.stderr, 'lstmscript.run: [info] loading dataset ... ', 
		
		eids_list = datica.load_config(opts.fname_config)
		dataset = datica.load_by_config(DIR_UNIGRAM, eids_list, datalen) 

		n_emo = len(eids_list)
		print >> sys.stderr, 'Done'

		print >> sys.stderr, 'lstmscript.run: [info] initialization of embedder'
		embedder = self.init_embedder(dataset, fname_embedder)

		print >> sys.stderr, 'lstmscript.run: [info] preparing input'
		dataset, Wemb = self.prepare_input(dataset, embedder)

		#################### Preparation for Output ############
		self.init_folder()
	
		#################### Training ##########################
		print >> sys.stderr, 'lstmscript.run: [info] start training'
	
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
		test_x, test_y = dataset[2]
		preds_prob = classifier.classify(test_x)
		cPickle.dump((test_y, preds_prob), open(fname_test, 'w'))

		###################### Report ############################
		validatica.report(test_y, preds_prob, DIR_TEST + prefix)

def main():
	script = LstmScriptRand()
	script.run()

if __name__ == '__main__':
	main()

