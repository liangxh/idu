#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.16
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
import blogger
from const import N_EMO, DIR_TEXT, DIR_UNIGRAM, DIR_TOKEN

def prepare(n_emo = N_EMO):
	'''
	tokenize and unigramize the text under data/dataset/text
	'''

	import blogger
	import zhtokenizer
	from utils import progbar, zhprocessor
	
	if not os.path.isdir(DIR_UNIGRAM):
		os.mkdir(DIR_UNIGRAM)

	if not os.path.isdir(DIR_TOKEN):
		os.mkdir(DIR_TOKEN)

	unigram_list = []
	token_list = []

	for eid in range(n_emo):
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

def load_data(dirname, n_emo, datalen = None, valid_rate = 0.2, test_rate = 0.1):
	'''
	load the dataset of EID in [0, emo) with datalen for each under dirname
	'''

	datalist = []

	for eid in range(n_emo):
		datalist.append(cPickle.load(open(dirname + '%d.pkl'%(eid), 'r')))
	
	n_samples = len(datalist[0])

	if datalen is not None and n_samples > datalen:
		n_samples = datalen

	n_valid = int(valid_rate * n_samples)
	n_test = int(test_rate * n_samples)
	n_train = n_samples - n_valid - n_test

	def build_dataset(idx_range):
		x = []
		y = []
		for i in idx_range:
			for eid in range(n_emo):
				if len(datalist[eid][i]) == 0:
					# this is a bug from zhtokenizer.tokenize, not solved now 
					continue

				x.append(datalist[eid][i])
				y.append(eid)
		return x, y

	train = build_dataset(range(n_train))
	valid = build_dataset(range(n_train, n_train + n_valid))
	test = build_dataset(range(n_samples - n_test, n_samples))

	return train, valid, test

def load_unigram(n_emo = N_EMO, datalen = None):
	'''
	load dataset under data/dataset/unigram
	'''
	return load_data(DIR_UNIGRAM, n_emo, datalen)

def load_token(n_emo = N_EMO, datalen = None):
	'''
	load dataset under data/dataset/token
	'''
	return load_data(DIR_TOKEN, n_emo, datalen)

if __name__ == '__main__':
	prepare()
