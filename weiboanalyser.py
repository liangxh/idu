#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.14
'''

import numpy as np
import weiboloader
import blogger

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

		for blog in blogs:
			total += blog['comments_count']
			for comm in blog['comments']:
				res = blogger.extract(comm['text'])
				if res == None or len(res[0]) < 2:
					continue
				count += 1
				tlen.append(len(res[0]))

		return 100. * count / total, np.mean(tlen)

	ryes = emo_rate(blogs_yes)
	print 'emo_rate_yes: ', ryes[0], '%, len: ', ryes[1]

	rno = emo_rate(blogs_no)
	print 'emo_rate_no: ', rno[0], '%, len: ', rno[1]

if __name__ == '__main__':
	main()
