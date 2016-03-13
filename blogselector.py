#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.11
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
import numpy as np

import commdatica
import blogger
from utils import progbar

def pkl_dump(d, fname):
	cPickle.dump(d, open(fname, 'w'))

def pkl_load(fname):
	return cPickle.load(open(fname, 'r'))

def export_unmv():
	import db
	con = db.connect()
	cur = con.cursor()

	cur.execute('select user_id, comments_count from microblogs where comments_count > 0')
	uc = {}
	for uid, cc in cur:
		if uc.has_key(uid):
			uc[uid].append(cc)
		else:
			uc[uid] = [cc, ]

	# u for user_id, n for n_blogs, m for mean(count), v for sqrt(var(count))
	unmv = [(uid, len(cc), np.mean(cc), np.sqrt(np.var(cc))) for uid, cc in uc.items()]
	pkl_dump(unmv, 'output/unmv.pkl')

	cur.close()
	con.close()

def select():
	unmv = pkl_load('output/unmv.pkl')

	import db
	con = db.connect()
	cur = con.cursor()

	thr_min = 5		

	# t for threshold_max
	unt = [(u, n, m + v) for u, n, m, v in unmv if m <= 50 and v <= 100]
	umtc = []
	for u, n, thr_max in unt:
		cur.execute('select mid, text, comments_count from microblogs where user_id = %s and comments_count >= %d and comments_count <= %d limit %d'%(u, thr_min, thr_max, n))

		tmp_umtc = []
		for m, t, c in cur:
			tmp_umtc.append((u, m, t, c))

		tmp_umtc = sorted(tmp_umtc, key = lambda k: -k[3])
		if len(tmp_umtc) > 100:
			tmp_umtc = tmp_umtc[:100]
		umtc.extend(tmp_umtc)
		
		if len(umtc) >= 400000:
			break

	fobj = open('output/umtc.txt', 'w')
	for u, m, t, c in umtc:
		fobj.write('%s\t%s\t%s\t%d\n'%(u, m, t, c))
	fobj.close()

def sample():
	blogs = commdatica.load('output/umtc.txt')
	
	has_emo = []
	no_emo = []

	target = 1000
	i = 0
	pbar = progbar.start(target)

	for blog in blogs:
		if blogger.is_valid(blog.text):
			if not len(has_emo) >= 500:
				has_emo.append(blog)
				i += 1
	
		elif blogger.is_valid(blog.text, check_emo = False):
			if not len(no_emo) >= 500:
				no_emo.append(blog)
				i += 1

		pbar.update(i)

	pbar.finish()

	print 'writing to umtc_yes_emo.txt ....',
	open('output/umtc_yes_emo.txt', 'w').write('\n'.join([repr(blog) for blog in has_emo]))
	print 'OK'

	print 'writing to umtc_no_emo.txt ....',
	open('output/umtc_no_emo.txt', 'w').write('\n'.join([repr(blog) for blog in no_emo]))
	print 'OK'	

	bs = commdatica.load('output/umtc_yes_emo.txt')
	print len(bs)

if __name__ == '__main__':
	#export_unmv()
	#select()
	sample()
