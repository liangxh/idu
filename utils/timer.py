#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.01.25
Modified: 2016.01.25
Description: a tool used to record the time consumed by a function
'''

import time
from functools import wraps

def timer():
	def wrapper(fn):
		@wraps(fn)
		def decorate(*args, **kwargs):
			st = time.time()
			ret = fn(*args, **kwargs)
			et = time.time()
			
			S = int(et - st)
			M = S / 60
			S %= 60
			H = M / 60
			M %= 60
			print 'timer: %s %02d:%02d:%02d'%(fn.__name__, H, M, S)
			return ret
		return decorate
	return wrapper