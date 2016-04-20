#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import datica

class CKClassifier:
	@classmethod
	def prepare_PMI_from_dataset(self, dataset):
		train, valid, test = dataset
		texts = []
		texts.extend(train[0])
		texts.extend(valid[0])
		texts.extend(test[0])
	
	@classmethod
	def prepare_PMI(self, texts):
		tf = {}
		thr = int(len(text) * 0.001)
		for tokens in texts:
			for token in tokens:
				if tf.has_key(token):
					tf[token] += 1
				else:
					tf[token] = 1

		print len(tf)
		print len([v for v in tf.values() if v >= thr])

def main():
	config = datica.load_config('data/config2.txt')
	dataset = datica.load_by_config('data/dataset/unigram', config)

	classifier = CKClassifier()
	classifier.prepare_PMI_from_dataset(dataset)

if __name__ == '__main__':
	main()

	
