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
import lstm
import datica
from const import N_EMO, DIR_EMBEDDER, DIR_TEST, DIR_MODEL

from lstm import LstmClassifier

def main():
	def init_embedder(prefix = 'id', dataset):
		'''
		initialize the embedder by load it from file if available
		or build the model by the dataset and save it
		'''

		if not os.path.isdir(DIR_EMBEDDER):
			os.mkdir(DIR_EMBEDDER)		
	
		fname = DIR_EMBEDDER + '%s_embedder.pkl'%(prefix)

		if os.path.exists(fname):
			print >> sys.stderr, 'model %s found and loaded'%(fname)
			return IdEmbedder.load(fname)
		else:
			def x_iterator(dataset):
				for set_x, set_y in dataset
					for x in set_x:
						yield x

			embedder = IdEmbedder()
			embedder.build(x_iterator(dataset))
			embedder.dump(fname)
	
		return embedder

	def embed_dataset(embedder, dataset):
		'''
		turn sequences of string into list of vectors
		'''

		def embed_set(set_x_y)
			x, y = set_x_y
			new_x = [embedder.embed(xi) for xi in x]
			return (new_x)		

		train, test, valid = dataset

		return (embed_set(train), embed_set(test), embed_set(valid)), embedder

	if not os.path.isdir(DIR_MODEL):
		os.path.mkdir(DIR_MODEL)

	if not os.path.isdir(DIR_TEST):
		os.path.mkdir(DIR_TEST)

	n_emo = N_EMO
	prefix = 'id'
	fname_model = DIR_MODEL + '%s_model.npz'%(prefix)
	fname_test = DIR_TEST + '%s_test.pkl'%(prefix)

	dataset = datica.load(n_emo)
	embedder = init_embedder(dataset, prefix)
	dataset = embed_dataset(embedder, dataset)

	classifier = LstmClassifier()
	res = classifier.train(
		dataset = dataset,
		ydim = n_emo,
		fname_model = fname_model,
	)

	test_x, test_y = dataset[2]
	if len(test_x[0][0]) == 1:
		test_x = [[[xi, xi] for xi in seq] for seq in test_x]

	preds_prob = classifier.classify(test_x)
	cPickle.dump((test_y, preds_prob), open(fname_test, 'w'))

if __name__ == '__main__':
	main()
from id_embedder import IdEmbedder
