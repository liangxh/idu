#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.09
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle

def senticov():
	emo_pa = cPickle.load(open('data/emodata/emo_pa.pkl', 'r'))
	emo_mids = cPickle.load(open('data/emo_mids.pkl', 'r'))
	emo_count = {}
	for emo, mids in emo_mids.items():
		emo_count[emo] = len(mids)

	n_total = sum(emo_count.values())
	phrases = [(emo, float(emo_count[emo])) for emo in emo_pa.keys()]

	count_sum = sum([c for p, c in phrases])
	phrases = sorted(phrases, key = lambda k:-k[1])
	phrases.append(('other', n_total - count_sum))

	for i in range(len(phrases)):
		p, c = phrases[i]
		phrases[i] = (p, c / n_total)

	fobj = open('data/senticov.txt', 'w')
	last = 0.
	for i, phrase in enumerate(phrases):
		p, c = phrase
		last += c
		fobj.write('%d. %6s\t(%.4f%%\t%.4f%%)\n'%(i + 1, p, c*100., last*100.))
	fobj.close()

def main():
	senticov()

if __name__ == '__main__':
	main()
