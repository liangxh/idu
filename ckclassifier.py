#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
from optparse import OptionParser


import datica
from utils import progbar


class CKClassifier:
	@classmethod
	def prepare_PMI_from_dataset(self, dataset, keep_rate):
		train, valid, test = dataset
		texts = []
		texts.extend(train[0])
		texts.extend(valid[0])
		texts.extend(test[0])

		self.prepare_PMI(texts, keep_rate)
	
	@classmethod
	def prepare_PMI(self, texts, keep_rate):
		tf = {}

		for tokens in texts:
			for token in tokens:
				if tf.has_key(token):
					tf[token] += 1
				else:
					tf[token] = 1

		n_samples = len(texts)
		n_token = int(len(tf) * keep_rate)

		tf = sorted(tf.items(), key = lambda k: -k[1])[: n_tokens]
		
		tokens_valid = {}
		for i, item in enumerate(tf):
			t, f = item
			tokens_valid[t] = i

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
						p[t1][t2] += 1
			
			l += 1
			pbar.update(l)
		pbar.finish()
		
		pmi_list = []
		values = []
		for i in range(n_tokens - 1):
			for j in range(i + 1, n_tokens):
				v = np.log2(p[i][j] * n_samples / (p[i], p[j]))
				pmi_list.append(((i, j), v))
				values.append(v)

		cPickle.dump(values, open('output/pmi_values.pkl', 'w'))

def main():
	optparser = OptionParser()
	optparser.add_option('-k', '--keep_rate', action='store', dest='keep_rate', type='str', default = 0.2)
	opts, args = optparser.parse_args()

	config = datica.load_config('data/config2.txt')
	dataset = datica.load_by_config('data/dataset/unigram/', config)

	classifier = CKClassifier()
	classifier.prepare_PMI_from_dataset(dataset, opts.keep_rate)

if __name__ == '__main__':
	main()

	
