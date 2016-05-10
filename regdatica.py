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

import cPickle
import numpy as np
import datica

from wordembedder import WordEmbedder
from utils import progbar

def prepare(key_embedder, ofname):
	fname_embedder = 'data/dataset/model/%s_embedder.pkl'%(key_embedder)
	embedder = WordEmbedder.load(fname_embedder)	

	train, valid, test = datica.load_data('data/dataset/unigram/', 90, valid_rate = 0.)
	
	def embed(xy):
		seqs, y = xy
		x_vec = []
		pbar = progbar.start(len(seqs))

		for i, seq in enumerate(seqs):
			x_vec.append(np.mean(embedder.embed(seq), axis = 0))
			pbar.update(i + 1)

		pbar.finish()

		return (x_vec, y)
	
	new_train = embed(train)
	new_test = embed(test)

	cPickle.dump((new_train, new_test), open(ofname, 'w'))

def get_sentiy(key_xvec, key_ofname):
	emo_pa = cPickle.load(open('data/emodata/emo_pa.pkl', 'r'))
	phrases = open('data/eid.txt', 'r').read().decode('utf8').split('\n')[:90]
	train, test = cPickle.load(open('data/dataset/xvec/%s.pkl'%(key_xvec), 'r'))
	
	def filter_eids(xy):
		x, y = xy
		new_x = []
		new_y = []
		for xi, yi in zip(x, y):
			p = phrases[yi]
			if emo_pa.has_key(p):
				new_x.append(xi)
				new_y.append(emo_pa[p])

		return new_x, new_y

	new_train = filter_eids(train)
	new_test = filter_eids(test)
	
	ofname = 'data/dataset/regdata/%s.pkl'%(key_ofname)
	cPickle.dump((new_train, new_test), open(ofname, 'w'))

def main():
	key_embedder = sys.argv[1]
	ofname = sys.argv[2]

	prepare(key_embedder, ofname)

if __name__ == '__main__':
	main()
