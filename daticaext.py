#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.13
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import cPickle

import blogger
from utils import progbar

def init_folders(dnames):
	for dname in dnames:
		if not os.path.isdir(dname):
			os.mkdir(dname)

def unigramize_dataset(dname_dataset, n_emo):
	'''
	prepare (unigramize and decompose above-follow) raw data from idname
	'''

	import zhtokenizer

	idname = 'data/blogs/dataset/raw/'
	odname_textu = 'data/blogs/dataset/text_unigram/'
	odname_aboveu = 'data/blogs/dataset/above_unigram/'
	odname_followu = 'data/blogs/dataset/follow_unigram/'

	init_folders([odname_textu, odname_aboveu, odname_followu])

	for eid in range(n_emo):
		ifname = idname + '%d.txt'%(eid)

		ofname_textu = odname_textu + '%d.pkl'%(eid)
		ofname_aboveu = odname_aboveu + '%d.pkl'%(eid)
		ofname_followu = odname_followu + '%d.pkl'%(eid)

		all_textu = []
		all_aboveu = []
		all_followu = []

		with open(ifname, 'r') as ifobj:
			for line in ifobj:
				blog = json.loads(line)

				all_textu.append(zhtokenizer.unigramize(blog['text']))

				aboveu = []
				for comm in blog['above']:
					text, emos = blogger.decompose(comm)
					aboveu.append((zhtokenizer.unigramize(text), emos))
				all_aboveu.append(aboveu)

				followu = []
				for comm in blog['follow']:
					text, emos = blogger.decompose(comm)
					followu.append((zhtokenizer.unigramize(text), emos))
				all_followu.append(followu)

		cPickle.dump(all_textu, open(ofname_textu, 'w'))
		cPickle.dump(all_aboveu, open(ofname_aboveu, 'w'))
		cPickle.dump(all_followu, open(ofname_followu, 'w'))


def main():
	dname_dataset = sys.argv[1]
	n_emo = int(sys.argv[2])

	unigramize_dataset(dname_dataset, n_emo)

if __name__ == '__main__':
	main()

