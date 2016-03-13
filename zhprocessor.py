#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.09
Description: interface for using jieba
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re

import jieba
from jieba import posseg

jieba.initialize()
jieba.enable_parallel(4)

re_zh = u'\u4e00-\u9fa5'
pattern_not_zh = re.compile(u'[^%s]+'%(re_zh))

def extract(text):
	try:
		text = text.decode('utf8')
	except:
		print '[Warning] cannot convert to UTF-8: ', text
	
	text = re.sub(pattern_no_zh, ' ', text)
	return text

def segment(text, pos_tagging = False):
	if pos_tagging:
		return posseg.cut(text)
	else:
		return jieba.cut(text)

def tokenize(text):
	words = segment(text, True)
	tokens = []
	for w in words:
		if not w.flag == 'x':
			tokens.append(w.word)

	return tokens

if __name__ == '__main__':
	text = u'还我八号风球！！！！八号风球挂了一个晚上，偏偏要上班的时候没有了！！！今天还要上班！！！噩耗！！！[泪][泪][泪]'

	words = segment(text)
	print '/'.join(words)
	print

	words = segment(text, True)
	print '/'.join(['%s(%s)'%(w.word, w.flag) for w in words])
	print
	
	tokens = tokenize(text)
	print '/'.join(tokens)
	print

