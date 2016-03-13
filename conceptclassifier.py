#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Reference: Hierarchical structures induce long-range dynamical correlations in written texts
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import math
import numpy as np

def frequency(ls):
	freq = {}
	for l in ls:
		if freq.has_key(l):
			freq[l] += 1
		else:
			freq[l] = 1

	return freq

class ConceptClassifier:
	def __init__(self):
		self.load_stopwords()
	
	def load_stopwords(self):
		self.stopwords = set()

	def analyse(self,	
			tokens,
			d = 2,
			window_size = 200,
			calculate_freq_threshold = lambda L, a: 0.4 * math.sqrt(float(L) / a),
		):

		L = len(tokens)
		print 'L = ', L
		
		print 'filtering valid tokens...'
		# filter the valid tokens
		freq = frequency(tokens)
		freq = sorted(freq.items(), key = lambda k:-k[1])
		
		f_thr = calculate_freq_threshold(L, window_size)

		valid_tokens = []
		tid = {}
		d_thr = 0 # number of valid tokens
		for token, f in freq:
			if f < f_thr:
				break

			if not token in self.stopwords:
				valid_tokens.append((token, f))
				tid[token] = d_thr
				d_thr += 1

		# in case d < d_thr
		d = min(d, d_thr)

		print 'transfering list of tokens of list of ids...'
		# transfer list of tokens to list of ids
		tids = []
		for token in tokens:
			if not tid.has_key(token):
				tids.append(None)
			else:
				tids.append(tid[token])

		print 'initialize matrix R (the upper right corner)'
		# initialize matrix R (the upper right corner)
		mat_R = np.zeros((d_thr, d_thr), dtype = float)

		half_window_size = window_size / 2
		for i, tid_i in enumerate(tids[:-1]):
			if tid_i == None:
				continue
			
			for tid_j in tids[i + 1: min(i + half_window_size, L)]:
				if tid_j == None:
					continue
				
				if tid_i > tid_j:
					tid_i, tid_j = (tid_j, tid_i)
				mat_R[tid_i][tid_j] += 1

		print 'initialize matrix M (the upper right corner)'
		# initialize matrix M (the upper right corner)
		mat_M = np.zeros((d_thr, d_thr), dtype = float)
		for i in range(d_thr - 1):
			for j in range(i + 1, d_thr):
				tmp = float(window_size) * valid_tokens[i][1] * valid_tokens[j][1] / L 
				mat_M[i][j] = tmp
		
		print 'calculate matrix N'
		# calculate matrix N
		mat_N = np.zeros((d_thr, d_thr), dtype = float)
		for i in range(d_thr):
			for j in range(i + 1, d_thr):
				if mat_R[i][j] == 0:
					tmp = 0
				else:
					tmp = (mat_M[i][j] - mat_R[i][j]) / math.sqrt(mat_R[i][j])
				mat_N[i][j] = tmp
				mat_N[j][i] = tmp

		print 'SVD'
		# SVD
		mat_S, mat_V, mat_DT = np.linalg.svd(mat_N)
		vecs = {}
		for i in range(d_thr):
			token, f = valid_tokens[i]
			vecs[token] = mat_S[i][:d]

		return vecs

def prepare():


def test():
	import cPickle
	import zhprocessor
	from utils import logger, progbar

	FNAME_TEXT = 'output/0.txt'
	PKL_TOKENS = 'data/tokens.pkl'

	if not os.path.exists(PKL_TOKENS):
		logger.debug('loading lines...')
		lines = open(FNAME_TEXT, 'r').read().split('\n')
		
		logger.debug('preparing toknes..')	
		tokens = []

		pbar = progbar.start(len(lines))
		for i, line in enumerate(lines):
			res = blogger.extract(line)
			if res == None:
				continue
			text, emo = res
			tokens.extend(zhprocessor.tokenize(text))
			pbar.update(i + 1)

		pbar.finish()
		
		cPickle.dump(tokens, open(PKL_TOKENS, 'w'))
	else:
		tokens = cPickle.load(open('data/tokens.pkl', 'r'))

	cclassifier = ConceptClassifier()
	logger.debug('analysing tokens..')
	
	d = 10
	vecs = cclassifier.analyse(tokens, d)
	print len(vecs)
	
	vecs = vecs.items()
	repr_vecs = [vecs[0] for i in range(d)]
	for vec in vecs[1:]:
		for i in range(d):
			if abs(repr_vecs[i][1][i]) < abs(vec[1][i]):
				repr_vecs[i] = vec
	
	for k, v in repr_vecs:
		print k, v

if __name__ == '__main__':
	test()
	
