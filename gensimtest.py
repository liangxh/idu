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


def test():
	import blogger
	import zhtokenizer
	import zhencoder

	text = u'我很高兴'.decode('utf8')
	tokens = zhtokenizer.tokenize(text)
	for t in tokens:
		print t, [t,]

	lines = open('data/blogs1000.txt', 'r').readlines()
	sents = []
	for line in lines:
		l = line.strip()
		res = blogger.extract(l, check_emo = False)
		if res == None:
			continue
		t = res[0]
		sents.append(zhtokenizer.tokenize(t))
		#sents.append(t)

	'''encoder = zhencoder.ZhEncoder()
	encoder.build_code(sents)
	sents = [[str(t) for t in encoder.encode(sent)] for sent in sents]
	print sents[0]
'''
	model = gensim.models.Word2Vec(sents, min_count = 10, size = 10, workers = 4)
	
	outfile = 'output/work2vec_model'
	model.save(outfile)	
	new_model = gensim.models.Word2Vec.load(outfile)

	#tokens = encoder.encode(text)
	tokens = zhtokenizer.tokenize(text)
	for t in tokens:
		try:
			vec = model[t]
		except:
			vec = None
		print t, vec

if __name__ == '__main__':
	test()
