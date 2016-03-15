#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.15
Description: Cooccurrence Matrix for short text (length < 200 )
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time

import math
import numpy as np

from abstract_embedder import AbstractEmbedder

def get_tf(seqs):
	token_tf = {}

	for seq in seqs:
		for token in seq:
			if token_tf.has_key(token):
				token_tf[token] += 1
			else:
				token_tf[token] = 1

	return token_tf	

class CooccurrenceEmbedder(AbstractEmbedder):
	default_value = None

	def build(self, seqs, dim, min_count = 10, window_size = 20):
		lengths = [len(seq) for seq in seqs]
		L = np.sum(lengths)  # length the dataset
		mean_L = L / len(seqs)

		a = min(window_size, mean_L)      # for easy coding only
	
		tf = get_tf(seqs)
		n_token = len(tf)

		# get representative words	
		#min_count = 0.01 * math.sqrt(float(L) / a)
		n_repr = len([None for t, f in tf.items() if f >= min_count])

		sorted_tf = sorted(tf.items(), key = lambda k: -k[1])
		tinfo = {}
		for i, item in enumerate(sorted_tf):
			t, f = item
			tinfo[t] = (i, f, f >= min_count) # idx, freq, bool_repr?
		
		# initialization of matrix R
		mat_R = np.zeros((n_repr, n_repr), dtype = float)

		a_half = a / 2
		for seq in seqs:
			for i, t1 in enumerate(seq):
				n_seq = len(seq)
				for t2 in seq[i + 1: min(i + a, n_seq)]:
					info1 = tinfo[t1]
					info2 = tinfo[t2]

					if info1[0] == info2[0]:
						continue
					elif info1[2] and info2[2]:
						mat_R[info2[0]][info1[0]] += 1
						mat_R[info1[0]][info2[0]] += 1
		
		# initialization of matrix M
		vec_tf = np.asmatrix([f for t, f in sorted_tf[:n_repr]])
		mat_M = float(a) * vec_tf.T * vec_tf / L
		

		# initialization of matrix N
		mat_N = np.divide(mat_M - mat_R, mat_R)
		mat_N[mat_N == np.inf] = 0.

		# turn the matrix into map of tokens and vectors
		self.code = {}
		for i, item in enumerate(sorted_tf[:n_repr]):
			t, f = item
			self.code[t] = np.asarray(mat_N[i, :])[0].tolist()

		self.set_default_value = [0. for i in range(n_repr)]

ClassEmbedder = CooccurrenceEmbedder

def test():
	import os
	import cPickle
	import zhtokenizer
	
	def load_sample():
		fname = 'data/blogs1000.pkl'
		if os.path.exists(fname):
			lines = cPickle.load(open(fname, 'r'))
		else:
			lines = open('data/blogs1000.txt', 'r').read().split('\n')
			lines = [zhtokenizer.tokenize(l) for l in lines]
			cPickle.dump(lines, open(fname, 'w'))
		
		return lines
	
	print >> sys.stderr, 'loading input data...', 
	lines = load_sample()
	print >> sys.stderr, 'done'

	embedder = ClassEmbedder()
	embedder.build(lines, 10)

	embedder.dump('output/1.pkl')

	ecd2 = ClassEmbedder.load('output/1.pkl')
	print ecd2.embed(lines[0])

if __name__ == '__main__':
	test()

