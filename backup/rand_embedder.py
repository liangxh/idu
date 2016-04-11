#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.14
Description: Random Embedder
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from abstract_embedder import AbstractEmbedder

import numpy as np

class RandEmbedder(AbstractEmbedder):
	''' 
	an embedder which represents each token by a randomly generated vector
	implemeneted according to DeepLearningTutorials.code.lstms
	'''
	default_value = None

	def build(self, seqs, dim):
		tokens = set()
		for seq in seqs:
			for token in set(seq):
				tokens.add(token)
		
		n_word = len(tokens)
		Wemb = 0.01 * np.random.rand(n_word, dim)
		
		self.code = {}
		for i, token in enumerate(tokens):
			self.code[token] = list(Wemb[i, :])

		self.set_default_value([0. for i in range(dim)])

ClassEmbedder = RandEmbedder

def test():		
	import zhtokenizer
	
	def load_sample():
		return open('data/blogs1000.txt', 'r').read().split('\n')

	lines = [zhtokenizer.tokenize(l) for l in load_sample()]
	
	embedder = ClassEmbedder()
	embedder.build(lines, 10)

	embedder.dump('output/1.pkl')

	ecd2 = ClassEmbedder.load('output/1.pkl')
	print ecd2.embed(lines[0])

if __name__ == '__main__':
	test()

