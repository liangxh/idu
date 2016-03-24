#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.24
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import cPickle

class InputAdapter:
	def __init__(self, Wemb = None, Widx = None):
		self.Wemb = Wemb
		self.Widx = Widx

	def build(self, embedder):
		Wemb = []
		Widx = {}

		Wemb.append(embedder.default_value) # Wemb[0] = default_value

		for i, item in enumerate(embedder.code.items()):
			k, v = item
			Widx[k] = i + 1			
			Wemb.append(v)

		self.Wemb = np.asarray(Wemb)
		self.Widx = Widx

	def represent_one(self, token_seq):
		return [self.Widx.get(token, 0) for token in token_seq]

	def represent(self, token_seqs):
		if not isinstance(token_seqs[0], list):
			token_seqs = [token_seqs, ]

		idx_seqs = []
		for token_seq in token_seqs:
			idx_seqs.append(self.represent_one(token_seq))

		return idx_seqs

	def get_Wemb(self,):
		return self.Wemb

	def dump(self, fname):
		cPickle.dump((self.Wemb, self.Widx), open(fname, 'w'))

	@classmethod
	def load(self, fname):
		Wemb, Widx = cPickle.load(open(fname, 'r'))
		return InputAdapter(Wemb, Widx)

if __name__ == '__main__':
	main()
