#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.22
Description: rebuilt-version of datica, new SL applied, new API added for different purpose
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import zhtokenizer

def prepare():	
	'''
	tokenize and unigramize the text under data/dataset/text
	'''
	import blogger
	import zhtokenizer
	from utils import progbar, zhprocessor
	
	dir_unigram = 'uniseq'
	dir_token = 'tokseq'

	if not os.path.isdir(DIR_UNIGRAM):
		os.mkdir(DIR_UNIGRAM)

	if not os.path.isdir(DIR_TOKEN):
		os.mkdir(DIR_TOKEN)

	unigram_list = []
	token_list = []

	for eid in eids:
		lines = open(DIR_TEXT + '%d.txt'%(eid), 'r').read().split('\n')
		
		unigram_list = []
		token_list = []
		
		print 'preparing data for EID-%d'%(eid)
		pbar = progbar.start(len(lines))
	
		for i, line in enumerate(lines):
			text, emo = blogger.extract(line)
			text = zhprocessor.simplify(text)

			unigrams = zhtokenizer.unigramize(text)	
			tokens = zhtokenizer.tokenize(text)
			
			unigram_list.append(unigrams)
			token_list.append(tokens)
		
			pbar.update(i + 1)
		pbar.finish()

		cPickle.dump(unigram_list, open(DIR_UNIGRAM + '%d.pkl'%(eid), 'w'))
		cPickle.dump(token_list, open(DIR_TOKEN + '%d.pkl'%(eid), 'w'))

