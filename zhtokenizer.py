#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.09
Description: functions for cutting sentences into unigrams or tokens, specifically for chinese based text
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re

import jieba
from jieba import posseg

jieba.initialize()
jieba.enable_parallel(4)

def segment(text, pos_tagging = False):
	'''
	interface for using jieba
	'''

	if pos_tagging:
		return posseg.cut(text)
	else:
		return jieba.cut(text)

def unigramize(text):
	'''
	turn 'hi, 你好' into ['hi', '你', '好']
	'''
	text = text.decode('utf8').lower()
	grams = []
		
	buf = ''
	for t in text:
		if t >= 'a' and t <= 'z':
			# merge the english words
			buf += t
			continue

		if not buf == '':
			grams.append(buf)
			buf = ''
			
		if not t == ' ':
			grams.append(t)

	return grams

def tokenize(text):
	'''
	turn 'hi, 你好' into ['hi', '你好']
	'''

	words = segment(text, True)
	tokens = []

	unwanted_pos = set(['x', 'm'])
	for w in words:
		if not w.flag in unwanted_pos:
			tokens.append(w.word.decode('utf8'))

	return tokens

if __name__ == '__main__':
	text = u'还我八号风球！！！！hello八号风球挂了一个晚上，偏偏要上班的时候没有了！！！今天还要上班！！！噩耗！！！[泪][泪][泪]'

	grams = unigramize(text)
	print '/'.join(grams)
	print 

	words = segment(text)
	print '/'.join(words)
	print

	words = segment(text, True)
	print '/'.join(['%s(%s)'%(w.word, w.flag) for w in words])
	print
	
	tokens = tokenize(text)
	print '/'.join(tokens)
	print

