#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.13
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from abstract_embedder import AbstractEmbedder

class IdEmbedder(AbstractEmbedder):
	'''
	give each token a unique index
	zero is preserved so indexing starts from 1
	'''

	default_value = 0

	def build(self, seqs):		
		self.n_code = 0
		self.code = {}
		for seq in seqs:
			for token in set(seq):
				if not self.code.has_key(token):
					self.n_code += 1
					self.code[token] = self.n_code

ClassEmbedder = IdEmbedder

def test():		
	import zhtokenizer
	
	def load_sample():
		return open('data/blogs1000.txt', 'r').read().split('\n')

	lines = [zhtokenizer.tokenize(l) for l in load_sample()]
	
	embedder = ClassEmbedder()	
	embedder.build(lines)

	embedder.dump('output/1.pkl')

	ecd2 = ClassEmbedder.load('output/1.pkl')
	print ecd2.embed(lines[0])

if __name__ == '__main__':
	test()
