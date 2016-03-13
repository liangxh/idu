#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.27
Description: a script used to analyse the distribution of comments_count in the whole database
'''

def tohist(ls):
	hist = {}
	for l in ls:
		if hist.has_key(l):
			hist[l] += 1
		else:
			hist[l] = 1
	return hist

def test():
	import cPickle
	import matplotlib.pyplot as plt

	ucommc = cPickle.load(open('output/ucommc.pkl', 'r'))
	
	uids = []
	commcs = []
	for u, commc in ucommc:
		uids.append(u)
		commcs.append(commc)

	uhist = tohist(uids)
	chist = tohist(commcs)

	plt.figure()
	plt.title('Distribution of Users')
	plt.xlabel('Frequency')
	plt.ylabel('Number of Users')
	plt.hist(uhist.values(), 50, alpha = 0.75)
	plt.savefig('output/userdist.png')

	plt.figure()
	plt.title('Distribution of comments_count')
	plt.xlabel('comments_count')
	plt.ylabel('Number of Blogs')
	plt.hist(chist.values(), 50, alpha = 0.75)
	plt.savefig('output/commcdist.png')


def analyse_uid():
	import db
	con = db.connect()
	cur = con.cursor()
	cur.execute('select ')

if __name__ == '__main__':
	test()
	pass

