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

def build(seqs, dim):
	tokens = set()
	for seq in seqs:
		tokens |= set(seq)

	n_word = len(tokens)	
	
	Widx = {}
	for i, token in enumerate(tokens):
		Widx[token] = i + 1
	
	Wemb = np.concatenate([np.zeros((1, dim)), 0.01 * np.random.rand(n_word, dim)], axis = 0)
	print Wemb.shape
	return Widx, Wemb

def test():
	import datica
	from wordembedder import WordEmbedder

	dataset = datica.load_unigram(2, 1000)
	train_x = dataset[0][0]
	wembedder = WordEmbedder(*build(train_x, 10))	
	print wembedder.embed(train_x[0])

if __name__ == '__main__':
	test()
