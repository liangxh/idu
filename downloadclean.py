#! /usr/env/python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.19
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import json
from optparse import OptionParser
from utils import progbar

def clean(ifname, ofname):
	lines = open(ifname, 'r').readlines()
	new_lines = []

	pbar = progbar.start(len(lines))
	for i, l in enumerate(lines):
		try:
			j = json.loads(l)
			new_lines.append(l)
		except ValueError:
			pass
		finally:
			pbar.update(i + 1)
	pbar.finish()

	fobj = open(ofname, 'w')
	fobj.write(''.join(new_lines))
	fobj.close()
	
	n_ori = len(lines)
	n_pass = len(new_lines)
	pass_rate = 100. * n_pass / n_ori
	print '%d / %d (%.2f%%)'%(n_pass, n_ori, pass_rate)
	

def main():
	parser = OptionParser()
	parser.add_option('-i', '--input', type='str', dest='ifile')
	parser.add_option('-o', '--output', type='str', dest='ofile')

	opts, args = parser.parse_args()

	clean(opts.ifile, opts.ofile)

if __name__ == '__main__':
	main()

