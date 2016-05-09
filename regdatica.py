#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.09
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import datica
from wordembedder import WordEmbedder

def prepare(key_embedder, ofname):
	fname_embedder = 'data/dataset/%s_embedder.pkl'%(key_embedder)
	embedder = WordEmbedder.load(fname_embedder)	

	train, valid, test = datica.load_data('data/dataset/unigram/', 90, valid_rate = 0.)
	
	def embed(xy):
		seqs, y = xy
		x_vec = [np.mean([embedder.embed(xi) for xi in seq]) for seq in seqs]
		return (x_vec, y)
	
	new_train = embed(train)
	new_test = embed(test)

	cPickle.dump((new_train, new_test), open(ofname, 'w'))

def main():
	key_embedder = sys.argv[1]
	ofname = sys.argv[2]

	prepare(key_embedder, ofname)

if __name__ == '__main__':
	main()
