#! /usr/env/python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.25
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle

import theano
import numpy as np
import gensim

def build(seqs, dim, min_count = 0, workers = 1):
	tokens = set()
	for seq in seqs:
		tokens |= set(seq)

	model = gensim.models.Word2Vec(seqs,
			size = dim,
			min_count = min_count,
			workers = workers
		)
	
	c = 0
	Widx = {}
	vecs = [[0. for i in range(dim)]]
	for token in tokens:
		try:
			vec = model[token]
		except KeyError:
			continue

		c += 1
		Widx[token] = c
		vecs.append(vec)

	Wemb = np.asarray(vecs).astype(theano.config.floatX)
	return Widx, Wemb

def test():
	import datica
	from wordembedder import WordEmbedder

	dataset = datica.load_unigram(2, 1000)
	train_x = dataset[0][0]
	wembedder = WordEmbedder(*build(train_x, 10))	
	print wembedder.embed(train_x[0])

if __name__ == '__main__':
	test()
