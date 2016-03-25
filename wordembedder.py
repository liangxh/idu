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

class WordEmbedder:
	default_value = 0

	def __init__(self, Widx = None, Wemb = None):
		self.Widx = Widx
		self.Wemb = np.asarray(Wemb).astype(theano.config.floatX)

	def index(self, seq):
		return [self.Widx.get(token, 0) for token in seq]

	def embed_one(self, token):
		'''
		return the vector representing the token
		return default_value when the token is not recognized
		'''
		return self.Wemb[self.Widx.get(token, 0)]

	def embed(self, seq):
		'''
		return a sequence of tokens into a list of vectors
		'''
		return [self.embed_one(token) for token in seq]

	def get_Wemb(self):
		return copy.deepcopy(self.Wemb)

	def dump(self, fname):
		'''
		save the class/model
		'''
		cPickle.dump((self.Widx, self.Wemb), open(fname, 'w'))

	@classmethod
	def load(self, fname):
		'''
		load a trained class/model
		'''
		Widx, Wemb = cPickle.load(open(fname, 'r'))
		return WordEmbedder(Widx, Wemb)

	def dimreduce_prepare(self):
		'''
		specific for dimreducer
		'''
		return self.Wemb[1:, :]

	def dimreduce_load(self, new_Wemb):
		'''
		specific for dimreducer
		'''
		self.Wemb = np.concatenate([np.zeros((1, new_Wemb.shape[1])), np.asarray(new_Wemb)], axis = 0).astype(theano.config.floatX)

	def dimreduce_fn(self, fn, *args, **kargs):
		self.dimreduce_load(fn(self.dimreduce_prepare(), *args, **kargs))

if __name__ == '__main__':
	pass
		
		
