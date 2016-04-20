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
import time

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

	st_time = time.time()
	print >> sys.stderr, 'executing... ',
	cur.execute('SELECT text FROM microblogs')
	print >> sys.stderr, time.time() - st_time
	
	pbar = progbar.start(all_N)
	l = 0

	for t in cur:
		res = blogger.extract(t[0], check_emo = False)
		if res is None:
			continue
	
		t = res[0]		
		pid = None
		p = None
		for i, pi in enumerate(patterns):
			if pi.search(t) >= 0:
				p = pi
				pid = i
				break
		
		if p is None:
			continue

		
		text = p.sub('', t)
		if counts[pid] < N:
			counts[pid] += 1
			fobjs[pid].write(text + '\n')

			l += 1
			pbar.update(l)

			if counts[pid] == N and sum(counts) == all_N:
				break

	pbar.finish()

	cur.close()
	con.close()

	print counts

	for fobj in fobjs:
		fobj.close()

if __name__ == '__main__':
	main()

