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

from utils import progbar

emos_list = [
	u'抓狂,怒,哼,怒骂'.split(','),
	u'鄙视,左哼哼,右哼哼,吐'.split(','),
	u'耶,哈哈,鼓掌,偷笑'.split(','),
	u'衰,生病,可怜,委屈'.split(','),
]

def main():
	import db
	import blogger
	from utils import progbar

	n_label = len(emos_list)
	emo_map = {}
	for label, emos in enumerate(emos_list):
		for emo in emos:
			emo_map[emo] = label

	odname = 'data/dataset_emo/'
	if not os.path.isdir(odname):
		os.mkdir(odname)

	odname += 'raw/'
	if not os.path.isdir(odname):
		os.mkdir(odname)


	fobjs = [open(odname + '%d.txt'%(i), 'w') for i in range(n_label)]
	counts = [0 for i in range(n_label)]
	N = 70000
	all_N = N * n_label

	con = db.connect()
	cur = con.cursor()
	
	pbar = progbar.start(all_N)
	l = 0

	cur.execute('SELECT text FROM microblogs')
	for t in cur:
		res = blogger.extract(t[0])
		if res is None or not emo_map.has_key(res[1]):
			continue

		label = emo_map(res[1])
		if counts[label] < N:
			counts[label] += 1
			fobjs[label].write(res[0] + '\n')

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

