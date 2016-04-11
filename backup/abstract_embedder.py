#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.14
Description: Abstract Embedder for inheritance
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle

class AbstractEmbedder:
	'''
	an abstract class inherited by IdEmbedder, RandEmbedder and so on
	'''
	default_value = 0

	def __init__(self):
		pass

	def build(self, seqs):
		'''
		input dataset of cut text to build the model
		model is trained and saved
		'''
		raise NotImplementedError

	def set_default_value(self, default_value):
		'''
		set the default value for unrecognized token
		'''

		self.default_value = default_value

	def get(self, token):
		'''
		return the vector representing the token
		return default_value when the token is not recognized
		'''
		return self.code.get(token, self.default_value)

	def embed(self, seq):
		'''
		return a sequence of tokens into a list of vectors
		'''
		return [self.get(token) for token in seq]

	def dump(self, fname):
		'''
		save the class/model
		'''
		cPickle.dump(self, open(fname, 'w'))

	@classmethod
	def load(self, fname):
		'''
		load a trained class/model
		'''
		return cPickle.load(open(fname, 'r'))

if __name__ == '__main__':
	pass
