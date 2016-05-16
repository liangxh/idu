#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.16
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
from optparse import OptionParser

def main():
	optparser = OptionParser()
	optparser.add_option('-i', '--input', action = 'store', dest = 'input', type = 'str')
	optparser.add_option('-n', '--topN', action = 'store', dest = 'topN', type = 'int', default = 10)
	opts, args = optparser.parse_args()

	prec = cPickle.load(open('data/dataset/test/%s_prec.pkl'%(opts.input), 'r'))
	print '  '.join(['(%d)%.4f'%(i + 1, prec[i]) for i in range(opts.topN)])

if __name__ == '__main__':
	main()
