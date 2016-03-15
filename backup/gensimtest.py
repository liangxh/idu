#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.14
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import gensim


def sample_seqs():
	import blogger
	import zhtokenizer
	
	lines = open('data/blogs1000.txt', 'r').readlines()
	for line in lines:
		l = line.strip()
		res = blogger.extract(l, check_emo = False)
		if res == None:
			continue
		t = res[0]
		yield zhtokenizer.tokenize(t)

def test():
	model = gensim.models.Word2Vec(sample_seqs(), min_count = 10, size = 10, workers = 4)
	
	outfile = 'output/work2vec_model'
	model.save(outfile)	
	new_model = gensim.models.Word2Vec.load(outfile)

	tokens = zhtokenizer.tokenize(text)
	for t in tokens:
		try:
			vec = model[t]
		except:
			vec = None
		print t, vec
	
if __name__ == '__main__':
	test()
