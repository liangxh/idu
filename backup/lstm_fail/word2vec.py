#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.15
Description: Word2Vec Embedder
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import gensim
from abstract_embedder import AbstractEmbedder

class Word2Vec(AbstractEmbedder):
	'''
	an embedder implemented based on Word2Vec
	'''

	default_value = None

	def build(self, seqs, dim, min_count = 0, workers = 4):
		'''
		simply call gensim.models.Word2Vec to construct the model 
		'''
		self.model = gensim.models.Word2Vec(seqs,
				size = dim, min_count = min_count, workers = workers)

		self.set_default_value([0. for i in range(dim)])

	def get(self, token):
		'''
		gensim.models.Word2Vec raise KeyError when $token is not recognized
		'''
		try:
			return list(self.model[token])
		except KeyError:
			return self.default_value

ClassEmbedder = Word2Vec

def test():		
	import zhtokenizer

	embedder = ClassEmbedder()
	def load_sample():
		return open('data/blogs1000.txt', 'r').read().split('\n')

	lines = [zhtokenizer.tokenize(l) for l in load_sample()]
	
	embedder.build(lines, 10)

	embedder.dump('output/1.pkl')
	embedder = ClassEmbedder.load('output/1.pkl')

	print embedder.embed(lines[0])

if __name__ == '__main__':
	test()


