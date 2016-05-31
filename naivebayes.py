#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.14
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import cPickle
from optparse import OptionParser

from utils import progbar

class ChiSquareSelector:
	@classmethod
	def calculate(self, x, y, deduplicate = True):
		vecs = {}
		ydim = np.max(y) + 1

		y_count = np.zeros(ydim)
		n_samples = len(x)

		for xi, yi in zip(x, y):
			y_count[yi] += 1

			if deduplicate:
				xi = set(xi)

			for token in xi:
				if not vecs.has_key(token):
					vecs[token] = np.zeros(ydim)
		
				vecs[token][yi] += 1

		chis = {}

		pbar = progbar.start(len(vecs))
		l = 0

		for token, vec in vecs.items():
			chi_values = []
		
			vec_sum = np.sum(vec)

			for i in range(ydim):
				a = vec[i]
				b = vec_sum - a
				c = y_count[i] - a
				d = (n_samples - y_count[i]) - b

				chi_values.append((a * d - b * c) ** 2 / ((a + b) * (c + d)))

			chis[token] = chi_values
			l += 1
			pbar.update(l)
		pbar.finish()

		return chis

	@classmethod
	def select(self, x, y, n, stopwords = set()):
		chis = self.calculate(x, y)
		
		ydim = np.max(y) + 1
		board = []
		for i in range(ydim):
			tokens = sorted(chis.items(), key = lambda k: -k[1][i])
			if len(tokens) > n:
				tokens = tokens[:n]
			tokens = [t[0] for t in tokens]
			board.append(tokens)

		feats = set()
		c = 0
		idx = 0
		
		pbar = progbar.start(n)

		n_token = len(board[0])

		while c < n and idx < n_token:
			for i in range(ydim):
				token = board[i][idx]
				if token in stopwords:
					continue

				if not token in feats:
					feats.add(token)
					c += 1
					
					pbar.update(c)
					if c == n:
						break
			idx += 1

		pbar.finish()

		return feats

class NaiveBayesClassifier:
	def __init__(self):
		pass

	def train(self, x, y, k = 1., deduplicate = False, valid_tokens = None, stopwords = set()):
		check_valid = valid_tokens is not None
		check_stopword = len(stopwords) == 0		

		self.ydim = np.max(y) + 1
		self.valid_tokens = valid_tokens
		self.check_valid = check_valid

		n_y = np.zeros(self.ydim)
		n_y_len = np.zeros(self.ydim)
		n_samples = len(x)

		print >> sys.stderr, 'scaning train dataset'

		
		pbar = progbar.start(n_samples)
		l = 0
		count = {}
		for seq, yi in zip(x, y):
			if deduplicate == True:
				seq = set(seq)
			
			n_y[yi] += 1
			c = 0
			for token in seq:
				if check_valid:
					if not token in valid_tokens:
						c += 1
						continue
				elif check_stopword and token in stopwords:
					c += 1
					continue

				if not count.has_key(token):
					count[token] = np.zeros(self.ydim)
				count[token][yi] += 1

			n_y_len[yi] += len(seq) - c
	
			l += 1
			pbar.update(l)

		pbar.finish()

		print >> sys.stderr, 'normalization'

		self.prob_token = {}
		n_V = len(count)

		pbar = progbar.start(len(count))
		l = 0
		for t, c in count.items():
			#self.prob_token[t] = (c + k) / ((k + 1) * n_y_len)
			self.prob_token[t] = np.log((c + k) / ((k + 1) * n_y_len))

			l += 1
			pbar.update(l)

		pbar.finish()

		self.prob_global = n_y

	def classify(self, seq, p_global = None, show = False):
		if p_global is None:
			p_global = self.prob_global

		p = np.ones(self.ydim)
		for t in seq:
			if self.check_valid and not t in self.valid_tokens:
				if show:
					print '%20s: INVALID TOKEN'%(t)
				continue

			if not self.prob_token.has_key(t):
				if show:
					print '%20s: UNKNOWN'%(t)
				continue

			if show:
				print '%20s: '%(t), self.prob_token[t].tolist()

			#p *= self.prob_token[t]
			p += self.prob_token[t]

		p += np.log(p_global)

		#p *= p_global
		#p_sum = np.sum(p)
		#if p_sum > 0:
		#	p /= p_sum

		return p

	def dump(self, ofname):
		cPickle.dump(self, open(ofname, 'w'))

	@classmethod
	def load(self, ifname):
		return cPickle.load(open(ifname, 'r'))
	

def test(k, flag_half, chisquare, flag_stopword):
	datalist = cPickle.load(open('nb20160316.pkl', 'r'))

	if flag_half:
		np.random.shuffle(datalist[0])
		datalist[0] = datalist[0][:len(datalist[1])]

	y_n_samples = [len(dlist) for dlist in datalist]
	print '[', ', '.join(['%d:%d'%(i, n) for i, n in enumerate(y_n_samples)]) ,']'	

	ydim = len(datalist)
	train_rate = 0.7

	train = ([], [])
	train_x, train_y = train

	test = ([], [])
	test_x, test_y = test

	for y, dlist in enumerate(datalist):
		n_samples = len(dlist)
		n_train = int(n_samples * train_rate)
		n_test = n_samples - n_train

		train_x.extend(dlist[:n_train])
		train_y.extend([y for i in range(n_train)])

		test_x.extend(dlist[n_train:])
		test_y.extend([y for i in range(n_test)])

	if flag_stopword:
		import nltk
		stopwords = set(nltk.corpus.stopwords.words('english'))
	else:
		stopwords = set()

	if chisquare > 0:
		valid_tokens = ChiSquareSelector.select(train_x, train_y, chisquare, stopwords)
	else:
		valid_tokens = None

	classifier = NaiveBayesClassifier()
	classifier.train(train_x, train_y, k, True, valid_tokens, stopwords)

	test_record = np.asarray([np.zeros(2) for i in range(ydim)])
	for xi, yi in zip(test_x, test_y):
		p = classifier.classify(xi)
		p_yi = np.argmax(p)
		test_record[yi][1 if yi == p_yi else 0] += 1

	for i in range(ydim):
		record = test_record[i]
		print 'precision for class %d: %.2f%%'%(i, 100. * record[1] / np.sum(record))
	
	sum_record = np.sum(test_record, axis = 0)
	print 'precision: overall: %.2f%%'%(100. * sum_record[1] / np.sum(sum_record))

def main():
	optparser = OptionParser()
	optparser.add_option('-k', '--value_k', dest='value_k', type='float', action = 'store', default = 1.)
	optparser.add_option('-f', '--half', dest='flag_half', action = 'store_true', default = False)
	optparser.add_option('-c', '--chisquare', dest='chisquare', action='store', type = 'int', default = 0)
	optparser.add_option('-s', '--stopword', dest='flag_stopword', action='store_true', default = False)
	opts, args = optparser.parse_args()

	test(opts.value_k, opts.flag_half, opts.chisquare, opts.flag_stopword)

if __name__ == '__main__':
	main()
