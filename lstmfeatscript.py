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
from const import N_EMO, DIR_MODEL, DIR_TEST
from lstm import LstmClassifier
from wordembedder import WordEmbedder

class LstmFeatScript:
	def __init__(self):
		self.init_default_options()
		self.add_extra_options()
		
	def init_default_options(self):
		parser = OptionParser()

		# necessary
		parser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
		
		# optional
		parser.add_option('-u', '--unigram', action='store_true', dest='unigram', default = False)
		# parser.add_option('-r', '--resume', action='store_true', dest='resume', default = False)

		# debug
		parser.add_option('-y', '--ydim', action='store', type='int', dest='ydim', default = N_EMO)
		parser.add_option('-n', '--n_samples', action='store', dest='n_samples', default = None)

		# especially for gpu
		# parser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)

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
			print >> sys.stderr, 'embedding model %s not found'%(fname_embedder)
			return None

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

		n_emo = opts.ydim
		datalen = opts.n_samples

		prefix = opts.prefix
		fname_model = DIR_MODEL + '%s_model.npz'%(prefix)
		fname_embedder = DIR_MODEL + '%s_embedder.pkl'%(prefix)
		fname_feat = DIR_MODEL + '%s_feat.pkl'%(prefix)

		print >> sys.stderr, 'Done'

		#################### Preparation of Input ##############
		print >> sys.stderr, 'lstmscript.run: [info] loading dataset ... ', 

		if opts.unigram:
			dataset = datica.load_unigram(n_emo, datalen) 
		else:
			dataset = datica.load_token(n_emo, datalen)

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

		if not os.path.exists(fname_model):
			print >> sys.stderr, 'model %s not found'%(fname_model)
			return
		else:
			print >> sys.stderr, 'model %s found and loaded'%(fname_model)
			classifier.load(fname_model)

		###################### Feature Extraction ##############################
		

		train, valid, test = dataset
		
		def tofeat(set_x_y):
			x, y = set_x_y
			new_x = classifier.classify(x)
			return (new_x, y)

		featset = (tofeat(train), tofeat(valid), tofeat(test))
				
		cPickle.dump(featset, open(fname_feat, 'w'))

def main():
	script = LstmFeatScript()
	script.run()

if __name__ == '__main__':
	main()

