#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.14
Description: analyse the sample comments
'''

import numpy as np
import weiboloader
import blogger

def tohist(ls):
	hist = {}
	for l in ls:
		if hist.has_key(l):
			hist[l] += 1
		else:
			hist[l] = 1
	return hist

def main():
	blogs_yes = weiboloader.load('output/comm_yes_emo.txt')[:500]
	blogs_no = weiboloader.load('output/comm_no_emo.txt')[:500]

	mean_cc_yes = np.mean([blog['comments_count'] for blog in blogs_yes])
	mean_cc_no = np.mean([blog['comments_count'] for blog in blogs_no])

	print 'mean_cc_yes: ', mean_cc_yes
	print 'mean_cc_no: ', mean_cc_no

	def emo_rate(blogs):
		count = 0
		total = 0

		tlen = []
		emolist = []

		for blog in blogs:
			total += blog['comments_count']
			for comm in blog['comments']:
				res = blogger.extract(comm['text'])
				if res == None or len(res[0]) < 2:
					continue
				count += 1
				tlen.append(len(res[0]))
				emolist.append(res[1])
		emohist = tohist(emolist)

		return 100. * count / total, np.mean(tlen), emohist

	def write_emo(fname, emohist):
		fobj = open(fname, 'w')
		emohist = sorted(emohist.items(), key = lambda k:-k[1])
		for i, item in enumerate(emohist):
			k, v = item
			fobj.write('%d. %s (%d)\n'%(i + 1, k, v))
		fobj.close()

	ryes = emo_rate(blogs_yes)
	print 'emo_rate_yes: %.2f%%, len: %.2f, total_emo: %d'%(ryes[0], ryes[1], np.sum(ryes[2].values()))
	write_emo('output/emo_yes.txt', ryes[2])

	rno = emo_rate(blogs_no)
	print 'emo_rate_no: %.2f%%, len: %.2f, total_emo: %d'%(rno[0], rno[1], np.sum(rno[2].values()))
	write_emo('output/emo_no.txt', rno[2])

if __name__ == '__main__':
	main()
