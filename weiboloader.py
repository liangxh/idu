#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.08
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import json

from weibolauncher import JSONS_COMMENT

def load(fname = JSONS_COMMENT):
	lines = open(fname, 'r').readlines()
	blogs = [json.loads(line) for line in lines]

	return blogs

def showinfo(blog):
	for k, v in blog.items():
		if not k == 'comments':
			print k, v
		else:
			print 'comments:'
			for com in v:
				print '\t %s to %s: %s'%(com['from_name'], com['to_name'], com['text'])

def test(fname = JSONS_COMMENT):
	blogs = load(fname)
	print len(blogs)
	blog = blogs[0]
	for k, v in blog.items():
		if not k == 'comments':
			print k, v
		else:
			print 'comments:'
			for com in v:
				print '\t %s to %s: %s'%(com['from_name'], com['to_name'], com['text'])

if __name__ == '__main__':
	test('data/launchtest.txt')
	
