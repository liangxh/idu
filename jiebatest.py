#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.17
Description: a script to test jieba for knowing which word has the flag 'x' or 'm'
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import zhtokenizer as zht
from utils import zhprocessor as zhp
from utils import progbar

def main():
	dlist = []
	x_set = {}
	m_set = {}

	for i in range(90):
		fname = 'data/dataset/text/%d.txt'%(i)
		lines = open(fname, 'r').read().split('\n')
		dlist.append(lines)		

		pbar = progbar.start(len(lines))
		for j, line in enumerate(lines):
			l = zhp.simplify(line.decode('utf8'))
			res = zht.segment(l, True)	
			for w in res:
				if w.flag == 'x':
					if x_set.has_key(w.word):
						x_set[w.word] += 1
					else:
						x_set[w.word] = 1
				elif w.flag == 'm':
					if m_set.has_key(w.word):
						m_set[w.word] += 1
					else:
						m_set[w.word] = 1
			pbar.update(j + 1)
		pbar.finish()

	fobj = open('output/set_x.txt', 'w')
	x_set = sorted(x_set.items(), key = lambda k: -k[1])
	for k, v in x_set:
		fobj.write('%s (%d)\n'%(k, v))
	fobj.close()

	fobj = open('output/set_m.txt', 'w')
	m_set = sorted(m_set.items(), key = lambda k: -k[1])
	for k, v in m_set:
		fobj.write('%s (%d)\n'%(k, v))
	fobj.close()

if __name__ == '__main__':
	main()

