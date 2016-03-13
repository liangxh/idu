#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.09
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import logging
import inspect

class Logger:
	def __init__(self, ostream = sys.stderr, level = logging.DEBUG):
		log_console = logging.StreamHandler(ostream)
		self.default_logger = logging.getLogger(__name__)
		self.default_logger.setLevel(level)
		self.default_logger.addHandler(log_console)
	
	@staticmethod
	def func_name(depth = 3):
		res = inspect.stack()
		if res[depth][0].f_locals.has_key('self'):
			'''
			function of a class
			'''
			return res[depth][0].f_locals['self'].__class__.__name__ + '.' + res[depth][3]
		else:
			'''
			function of a module 
			'''
			return res[depth][3]

	def log(self, level, levelname, msg):
		funcname = Logger.func_name()
		self.default_logger.log(level,
			'%(funcname)s: [%(levelname)s] %(message)s',
			{'levelname':levelname, 'funcname':funcname, 'message':msg}
		)

	def debug(self, msg):
		self.log(logging.DEBUG, 'Debug', msg)

	def info(self, msg):
		self.log(logging.INFO, 'Info', msg)

	def warning(self, msg):
		self.log(logging.WARNING, 'Warning', msg)

	def error(self, msg):
		self.log(logging.ERROR, 'Error', msg)

	def critical(self, critical):
		self.log(logging.CRITICAL, 'Critical', msg)

if __name__ == '__main__':
	logger = Logger()

	class TestClass:
		def funcA(self):
			logger.info('Hello World')

	test = TestClass()
	test.funcA()
