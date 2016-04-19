#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.19
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re

from utils import progbar

def main():
	import db
	import blogger
	from utils import progbar

	p1 = re.compile('[:;]\)')
	p2 = re.compile('[:;]\(')
	patterns = [p1, p2]
	n_label = 2
	N = 70000
	
	odname = 'data/dataset_sym/'
	if not os.path.isdir(odname):
		os.mkdir(odname)

	odname += 'raw/'
	if not os.path.isdir(odname):
		os.mkdir(odname)


	fobjs = [open(odname + '%d.txt'%(i), 'w') for i in range(n_label)]
	counts = [0 for i in range(n_label)]
	all_N = N * n_label

	con = db.connect()
	cur = con.cursor()


	print >> sys.stderr, 'executing... '
	cur.execute('SELECT text FROM microblogs LIMIT 500')
	
	pbar = progbar.start(all_N)
	l = 0

	for t in cur:
		t = t[0]
		pid = None
		p = None
		for i, pi in enumerate(patterns):
			if pi.search(t) >= 0:
				p = pi
				pid = i
				print t
				break
		
		if p is None:
			continue

		res = blogger.extract(t, check_emo = False)
		if res is None:
			continue

		text = p.sub('', res[0])
		if counts[label] < N:
			counts[pid] += 1
			fobjs[pid].write(text + '\n')

			l += 1
			pbar.update(l)

			if counts[label] == N and sum(counts) == all_N:
				break

	pbar.finish()

	cur.close()
	con.close()

	for fobj in fobjs:
		fobj.close()

if __name__ == '__main__':
	main()

