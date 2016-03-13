#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.26
Description: a script used to analyse the coverage the hownet
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle


from const import N_EMO, DIR_TEXT, DIR_TOKEN
from utils import progbar
import matplotlib.pyplot as plt

def test(n_emo = N_EMO):
	import tokenprocessor as tokenpr
	from hownet import HowNet

	hownet = HowNet()

	content_all = 0
	content_supported = 0	

	token_all = set()
	
	token_supported = {}
	
	pbar = progbar.start(n_emo * 4000)

	i = 0
	token_not_supported = {}
	coverdist = []

	eseqs = tokenpr.load(n_emo)
	for eid, seqs in enumerate(eseqs):
		for seq in seqs:
			content_all += len(seq)
			
			sup_len = 0
			for token in seq:
				if hownet.support(token):
					sup_len += 1
					content_supported += 1

					if token_supported.has_key(token):
						token_supported[token] += 1
					else:
						token_supported[token] = 1
				else:
					if token_not_supported.has_key(token):
						token_not_supported[token] += 1
					else:
						token_not_supported[token] = 1

			coverdist.append((len(seq), sup_len))

			i += 1
			pbar.update(i)

	pbar.finish()

	n_sup_token = len(token_supported.keys())
	n_nsup_token = len(token_not_supported.keys())
	n_token = n_sup_token + n_nsup_token

	print 'n_token: %d'%(n_token)
	print 'token_coverage: %.1f%%'%(100. * n_sup_token / n_token)
	
	print 'n_content: %d'%(content_all)
	print 'content_coverage: %.1f%%'%(100. * content_supported / content_all)

	def save1(fname, tfhist):
		fobj = open(fname, 'w')
		for i, item in enumerate(sorted(tfhist.items(), key = lambda k:-k[1])):
			t, tf = item
			fobj.write('%d. %s (%d)\n'%(i + 1, t, tf))
		fobj.close()

	save1('output/token_supported.txt', token_supported)
	save1('output/token_not_supported.txt', token_not_supported)

	cPickle.dump(coverdist, open('output/coverdist.pkl', 'w'))


def show_coverdist():
	coverdist = cPickle.load(open('output/coverdist.pkl', 'r'))

	plt.figure()
	x = [float(b) / a  for a, b in coverdist]
	n, bins, patches = plt.hist(x, 50, alpha=0.75)
	
	plt.title('Distribution of HowNet Coverage')
	plt.xlabel('Coverage')
	plt.ylabel('Number of Blogs')
	plt.savefig('output/covdist.png')

if __name__ == '__main__':
	#test()
	show_coverdist()
