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
from rand_embedder import RandEmbedder
from inputadapter import InputAdapter
from lstm import LstmClassifier

def main():
	parser = OptionParser()

	# necessary
	parser.add_option('-d', '--dim_proj', action='store', type = 'int', dest='dim_proj') # , default = 128
	parser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
	parser.add_option('-r', '--resume', action='store_true', dest='resume', default = False)

	# optional
	parser.add_option('-u', '--unigram', action='store_true', dest='unigram', default = False)

	# debug
	parser.add_option('-y', '--ydim', action='store', type='int', dest='ydim', default = N_EMO)
	parser.add_option('-n', '--n_samples', action='store', dest='n_samples', default = None)

	# especially for gpu
	parser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)

	opts, args = parser.parse_args()

	n_emo = opts.ydim
	datalen = opts.n_samples
	prefix = opts.prefix
	dim_proj = opts.dim_proj

	fname_model = DIR_MODEL + '%s_model.npz'%(prefix)
	fname_test = DIR_TEST + '%s_test.pkl'%(prefix)

	def init_embedder(dataset):
		'''
		initialize the embedder by load it from file if available
		or build the model by the dataset and save it
		'''
	
		fname = DIR_MODEL + '%s_embedder.pkl'%(prefix)

		if os.path.exists(fname):
			print >> sys.stderr, 'embedding model %s found and loaded'%(fname)
			return RandEmbedder.load(fname)
		else:
			def x_iterator(dataset):
				for set_x, set_y in dataset:
					for x in set_x:
						yield x

			embedder = RandEmbedder()
			embedder.build(x_iterator(dataset), dim_proj)
			embedder.dump(fname)
	
		return embedder

	def prepare_input(dataset, embedder):
		'''
		turn sequences of string into list of vectors
		'''
		
		adapter = InputAdapter()
		adapter.build(embedder)

		def represent_set(set_x_y):
			x, y = set_x_y
			new_x = adapter.represent(x)
			print len(new_x)
			return (new_x, y)

		train, test, valid = dataset
		new_dataset = (represent_set(train), represent_set(test), represent_set(valid))

		return new_dataset, adapter.get_Wemb()

	if not os.path.isdir(DIR_MODEL):
		os.mkdir(DIR_MODEL)

	if not os.path.isdir(DIR_TEST):
		os.mkdir(DIR_TEST)

	if opts.unigram:
		dataset = datica.load_unigram(n_emo, datalen)
	else:
		dataset = datica.load_token(n_emo, datalen)

	embedder = init_embedder(dataset)

	dataset, Wemb = prepare_input(dataset, embedder)
	
	classifier = LstmClassifier()

	if not os.path.exists(fname_model):
		res = classifier.train(
			dataset = dataset,
			Wemb = Wemb,
			ydim = n_emo,

			fname_model = fname_model,

			batch_size = opts.batch_size,
			valid_batch_size = opts.batch_size,
		)
	elif opts.resume:
		res = classifier.train(
			dataset = dataset,
			Wemb = Wemb,
			ydim = n_emo,

			fname_model = fname_model,
			reload_model = True,

			batch_size = opts.batch_size,
			valid_batch_size = opts.batch_size,
		)
	else:
		print >> sys.stderr, 'lstm model %s found and loaded'%(fname_model)
		classifier.load(fname_model)

	test_x, test_y = dataset[2]

	preds_prob = classifier.classify(test_x)
	cPickle.dump((test_y, preds_prob), open(fname_test, 'w'))

	validatica.report(test_y, preds_prob, DIR_TEST + prefix)

if __name__ == '__main__':
	main()

