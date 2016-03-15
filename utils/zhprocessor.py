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

from utils.langconv import Converter
sc_converter = Converter('zh-hans')

re_zh = u'\u4e00-\u9fa5'
pattern_not_zh = re.compile(u'[^%s]+'%(re_zh))

def simplify(text):
	'''
	turn traditional chinese to simplified chinese
	'''
	return sc_converter.convert(text)

def clean(text):
	'''
	not used, maybe altered and adjusted someday
	'''
	try:
		text = text.decode('utf8')
	except:
		print '[Warning] cannot convert to UTF-8: ', text
	
	text = re.sub(pattern_no_zh, ' ', text)
	return text

