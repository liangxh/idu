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
from const import N_EMO, DIR_TEXT, DIR_UNIGRAM, DIR_TOKEN, DIR_DATA

def prepare_unigramize(ifname, ofname):
	import zhtokenizer
	from utils import zhprocessor

	lines = open(ifname, 'r').readlines()

	seqs = []
	for line in lines:
		line = line.decode('utf8')
		line = zhprocessor.simplify(line)
		tokens = zhtokenizer.unigramize(line)
		seqs.append(tokens)

	cPickle.dump(seqs, open(ofname, 'w'))

def prepare(eids = range(N_EMO)):
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


def load_by_config(dirname, eids_list, datalen = None, valid_rate = 0.2, test_rate = 0.1):
	datalist = []

	for eids in eids_list:
		dlist = []
		for eid in eids:
			dlist.extend(cPickle.load(open(dirname + '%d.pkl'%(eid), 'r')))
		datalist.append(dlist)	

	n_samples = len(datalist[0])

	if datalen is not None and n_samples > datalen:
		n_samples = datalen

	n_valid = int(valid_rate * n_samples)
	n_test = int(test_rate * n_samples)
	n_train = n_samples - n_valid - n_test

	n_emo = len(datalist)

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

def load_config(ifname, fname_eid = DIR_DATA + 'eid.txt'):
	all_emos = open(fname_eid, 'r').read().decode('utf8').split('\n')

	emos_list = []
	fobj = open(ifname, 'r')

	ydim = 0
	emomap = {}
	n_emo = 0

	for line in fobj:
		emos = line.decode('utf8').strip().split(',')
		if len(line) == 0:
			break

		for emo in emos:
			emomap[emo] = ydim
		ydim += 1
		n_emo += len(emos)

	print n_emo

	config = [[] for i in range(ydim)]
	for eid, emo in enumerate(all_emos):
		if emomap.has_key(emo):
			config[emomap[emo]].append(eid)
			n_emo -= 1
			if n_emo == 0:
				break
	
	return config
	


if __name__ == '__main__':
	prepare()
