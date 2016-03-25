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

import math
import numpy as np
from utils import progbar

def get_tf(seqs):
	token_tf = {}

	for seq in seqs:
		for token in seq:
			if token_tf.has_key(token):
				token_tf[token] += 1
			else:
				token_tf[token] = 1

	return token_tf	

def build(seqs, min_count = 0, window_size = 20):
	lengths = [len(seq) for seq in seqs]
	L = np.sum(lengths)  # length the dataset
	mean_L = L / len(seqs)

	a = min(window_size, mean_L)      # for easy coding only
	
	# get representative words
	#min_count = 0.01 * math.sqrt(float(L) / a)
	
	tf = get_tf(seqs)
	tf = sorted(tf.items(), key = lambda k: -k[1])
	tinfo = {}
	for i, item in enumerate(tf):
		t, f = item
		if f < min_count:
			print t, f, min_count
		tinfo[t] = (i, f, f >= min_count) # idx, freq, bool_repr?

	n_repr = len([None for k, v in tinfo.items() if v[2]])

	# initialization of matrix R
	print >> sys.stderr, 'wemb_cooc.build: [info] initialization of matrix R'
	mat_R = np.zeros((n_repr, n_repr), dtype = float)

	a_half = a / 2
	
	pbar = progbar.start(len(seqs))

	for l, seq in enumerate(seqs):
		for i, t1 in enumerate(seq):
			n_seq = len(seq)
			for t2 in seq[i + 1: min(i + a, n_seq)]:
				info1 = tinfo[t1]
				info2 = tinfo[t2]

				if info1[0] == info2[0]:
					continue
				elif info1[2] and info2[2]:
					mat_R[info2[0]][info1[0]] += 1.
					mat_R[info1[0]][info2[0]] += 1.

		pbar.update(l + 1)
	pbar.finish()
		
	# initialization of matrix M
	print >> sys.stderr, 'wemb_cooc.build: [info] initialization of matrix M'
	
	vec_tf = np.asmatrix([f for t, f in tf[:n_repr]])
	mat_M = float(a) * vec_tf.T * vec_tf / L
	
	# initialization of matrix N
	print >> sys.stderr, 'wemb_cooc.build: [info] initialization of matrix N'

	mat_N = np.divide(mat_M - mat_R, mat_R)
	mat_N[mat_N == np.inf] = 0.
	
	Widx = {}
	for i, item in enumerate(tf[:n_repr]):
		t, f = item
		Widx[t] = i + 1
	
	Wemb = np.concatenate([np.zeros((1, n_repr)), mat_N], axis = 0)
	
	print >> sys.stderr, 'wemb_cooc.build: [info] finish'
	return Widx, Wemb

def test():
	import datica
	from wordembedder import WordEmbedder

	dataset = datica.load_unigram(2, 1000)
	train_x = dataset[0][0]
	wembedder = WordEmbedder(*build(train_x))	
	print wembedder.embed(train_x[0])

if __name__ == '__main__':
	test()
