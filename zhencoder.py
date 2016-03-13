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

from utils import progbar
from zhtokenizer import unigramize, tokenize

ENCODER_UNIGRAM = 1
ENCODER_TOKEN = 2

class ZhEncoder:
	def __init__(self, encoder = ENCODER_TOKEN):
		self.f_tokenize = {ENCODER_UNIGRAM: unigramize, ENCODER_TOKEN: tokenize}[encoder]
		self.n_code = 0
		self.code = {}

	def build_code(self, lines):
		def build_one(self, line):
			tokens = self.f_tokenize(line)

			for token in set(tokens):
				if not self.code.has_key(token):
					self.code[token] = self.n_code
					self.n_code += 1
		
		if not isinstance(lines, list):
			build_one(self, lines)
		else:
			for line in lines:
				build_one(self, line)

	def encode(self, line):
		tokens = self.f_tokenize(line)
		codes = [self.code.get(token, '-1') for token in tokens]

		return codes

	def dump(self, fname):
		cPickle.dump(self, open(fname, 'w'))

	@classmethod
	def load(self, fname):
		return cPickle.load(open(fname, 'r'))

def load_sample():
	return open('data/blogs1000.txt', 'r').read().split('\n')

def test():
	lines = load_sample()
	encoder = ZhEncoder()
	encoder.build_code(lines)

	encoder.dump('output/1.pkl')

	ecd2 = ZhEncoder.load('output/1.pkl')
	print ecd2.encode(lines[0])

if __name__ == '__main__':
	test()