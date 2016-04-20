#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
import numpy as np
from optparse import OptionParser


import datica
from utils import progbar

def dataset_to_texts(dataset):
	train, valid, test = dataset
	texts = []
	texts.extend(train[0])
	texts.extend(valid[0])
	texts.extend(test[0])
	return texts

def filter_valid_tokens(texts, rate):
	tf = {}

	for tokens in texts:
		for token in tokens:
			if tf.has_key(token):
				tf[token] += 1
			else:
				tf[token] = 1

	n_samples = len(texts)
	n_tokens = int(len(tf) * rate)

	tf = sorted(tf.items(), key = lambda k: -k[1])[: n_tokens]

	tokens_valid = {}
	for i, item in enumerate(tf):
		t, f = item
		tokens_valid[t] = i

	return tokens_valid

def get_valid_rate(texts, tokens_valid):
	c = 0
	valid_tokens = set(tokens_valid.keys())

	for tokens in texts:
		tokens = set(tokens)
		if len(tokens.intersection(valid_tokens)) == 0:
			c += 1

	print 'invalid coverage: %d (%.2f%%)'%(c, 100. * c / len(texts))

class CKClassifier:	
	@classmethod
	def prepare_PMI(self, texts, tokens_valid, thr):
		n_tokens = len(tokens_valid)
		n_samples = len(texts)

		p_margin = np.zeros(n_tokens)
		p = np.zeros((n_tokens, n_tokens))
		
		pbar = progbar.start(n_samples)
		l = 0

		for tokens in texts:
			tids = [tokens_valid[token] for token in set(tokens) if tokens_valid.has_key(token)]

			for tid in tids:
				p_margin[tid] += 1

			for i in range(len(tids) - 1):
				t1 = tids[i]
				for j in range(i + 1, len(tids)):
					t2 = tids[j]
					if t1 < t2:
						p[t1][t2] += 1
					else:
						p[t2][t1] += 1
			
			l += 1
			pbar.update(l)
		pbar.finish()
		
		pmi_list = []
		#values = []
	
		n = (n_tokens - 1) * n_tokens / 2
		pbar = progbar.start(n)
		l = 0

		for i in range(n_tokens - 1):
			if p_margin[i] == 0.:
					print i
				
			for j in range(i + 1, n_tokens):
				v = p[i][j] / (p_margin[i] * p_margin[j])

				if v > thr:
					pmi_list.append(((i, j), v))
				
				#values.append(v)
				l += 1
				pbar.update(l)
		pbar.finish()

		#cPickle.dump(values, open('output/pmi_values.pkl', 'w'))
		return pmi_list

def main():
	optparser = OptionParser()
	optparser.add_option('-k', '--keep_rate', action='store', dest='keep_rate', type='str', default = 0.2)
	opts, args = optparser.parse_args()

	###################### Load dataset ###################################
	config = datica.load_config('data/config2.txt')
	dataset = datica.load_by_config('data/dataset/unigram/', config)

	###################### Preparation ###################################
	texts = dataset_to_texts(dataset)


	thr_PMI = 0.02
	rate_TF = opts.keep_rate

	tokens_valid = filter_valid_tokens(texts, rate_TF)
	get_valid_rate(texts, tokens_valid)
	
	classifier = CKClassifier()
	pmi_list = CKClassifier.prepare_PMI(texts, tokens_valid, thr_PMI)


if __name__ == '__main__':
	main()

	
