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
import time
import theano
import numpy as np

from tfidfembedder import TfIdfEmbedder
from utils import progbar

def build(seqs, N):
	tf = {}
	df = {}

	pbar = progbar.start(len(seqs))
	l = 0

	for seq in seqs:
		for t in seq:
			if tf.has_key(t):
				tf[t] += 1
			else:	
				tf[t] = 1.
		
		for t in set(seq):
			if df.has_key(t):
				df[t] += 1
			else:
				df[t] = 1
			
		l += 1
		pbar.update(l)

	pbar.finish()

	tf = sorted(tf.items(), key = lambda k: -k[1])

	n_seq = len(seqs)
	
	nums = set('0123456789')
	Widx = {}
	Widf = {}
	idx = 0
	for t, f in tf:
		if not t in nums:
			Widf[t] = np.log((1. + n_seq) / df[t])
			Widx[t]	= idx	
			idx += 1
			if idx == N:
				break
	
	return Widx, Widf

def prepare_embedder():
	import datica
	train, valid, test = datica.load_data('data/dataset/unigram/', 90, valid_rate = 0.)

	x, y = train
	Widx = build(x, 2000)
	cPickle.dump(Widx, open('data/dataset/model/tfidf2000_embedder.pkl', 'w'))

def prepare_xvec():
	import datica
	train, valid, test = datica.load_data('data/dataset/unigram/', 90, valid_rate = 0.)

	embedder = TfIdfEmbedder.load('data/dataset/model/tfidf2000_embedder.pkl')

	st = time.time()
	def x2vec(xy):
		x, y = xy
		vecs = []
		for xi in x:
			vecs.append(embedder.embed(xi))

		return vecs, y

	train = x2vec(train)
	test = x2vec(test)

	print 'x2vec: %.2f sec'%(time.time() - st)

	#cPickle.dump((train, test), open('data/dataset/xvec/bow2000.pkl', 'w'))
	return train, test

if __name__ == '__main__':
	prepare_embedder()
	#prepare_xvec()
