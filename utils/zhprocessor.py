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

from utils.langconv import Converter
sc_converter = Converter('zh-hans')

def simplify(text):
	return sc_converter.convert(text)

