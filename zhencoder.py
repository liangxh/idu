#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.13
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle

import zhtokenizer
from utils import progbar

class ZhEncoder:
	def __init__(self):
		self.n_code = 0
		self.code = {}

	def build_code(self, seqs):
		def build_one(self, seq):
			for token in set(seq):
				if not self.code.has_key(token):
					self.code[token] = self.n_code
					self.n_code += 1
		
		if not isinstance(seqs[0], list):
			build_one(self, seqs)
		else:
			for seq in seqs:
				build_one(self, seq)

	def encode(self, seq):
		codes = [self.code.get(token, '-1') for token in seq]

		return codes

	def dump(self, fname):
		cPickle.dump(self, open(fname, 'w'))

	@classmethod
	def load(self, fname):
		return cPickle.load(open(fname, 'r'))

def load_sample():
	return open('data/blogs1000.txt', 'r').read().split('\n')

def test():
	lines = [zhtokenizer.tokenize(l) for l in load_sample()]
	encoder = ZhEncoder()
	encoder.build_code(lines)

	encoder.dump('output/1.pkl')

	ecd2 = ZhEncoder.load('output/1.pkl')
	print ecd2.encode(lines[0])

if __name__ == '__main__':
	test()
