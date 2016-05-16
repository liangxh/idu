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

import numpy as np
import theano
import copy

class BowEmbedder:
	def __init__(self, Widx = None):
		self.Widx = Widx
		self.N = len(self.Widx)	

	def embed(self, seq):
		'''
		return a sequence of tokens into a list of vectors
		'''
		vec = np.zeros(self.N)
		
		for t in seq:
			if self.Widx.has_key(t):
				vec[self.Widx[t]] += 1.

		vec_sum = vec.sum()
		if not vec_sum == 0.:
			vec /= vec_sum

		return vec

	def dump(self, fname):
		'''
		save the class/model
		'''
		cPickle.dump(self.Widx, open(fname, 'w'))

	@classmethod
	def load(self, fname):
		'''
		load a trained class/model
		'''
		Widx = cPickle.load(open(fname, 'r'))
		return BowEmbedder(Widx)

if __name__ == '__main__':
	pass
		
		
