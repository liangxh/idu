#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.16
'''
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from abstract_embedder import AbstractEmbedder

class CustomEmbedder(AbstractEmbedder):
	def build(self, code):
		dim = len(code.values()[0])
		self.code = code
		self.default_value = [0. for i in range(dim)]

def test():
	import os
	import cPickle
	import zhtokenizer
	import dimreducer
	from cooc_embedder import CooccurrenceEmbedder as CoocEmbedder

	def load_sample():
		fname = 'data/blogs1000.pkl'
		if os.path.exists(fname):
			lines = cPickle.load(open(fname, 'r'))
		else:
			lines = open('data/blogs1000.txt', 'r').read().split('\n')
			lines = [zhtokenizer.tokenize(l) for l in lines]
			cPickle.dump(lines, open(fname, 'w'))
		
		return lines
	
	print >> sys.stderr, 'loading input data...', 
	lines = load_sample()
	print >> sys.stderr, 'done'

	embedder = CoocEmbedder()
	embedder.build(lines)

	embedder.dump('output/1.pkl')
	embedder = CoocEmbedder.load('output/1.pkl')

	new_embedder = CustomEmbedder()
	new_embedder.build(dimreducer.svd(embedder.code, 10))
	#new_embedder.build(dimreducer.dA(embedder.code, 10))
	print new_embedder.embed(lines[0])

if __name__ == '__main__':
	test()
