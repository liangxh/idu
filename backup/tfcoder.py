#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.23
Description: set index for single character, not used anymore (maybe)
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle

class TfCoder:
	_n_code = None
	_code = {}
	_tf = {}
	_CODE_UNKNOWN = None

	def n_code(self):
		return self._n_code

	def code(self, text):
		tokens = self.tokenize(text)
		codes = []
		for t in tokens:
			if self._code.has_key(t):
				codes.append(self._code[t])
			else:
				codes.append(self._CODE_UNKNOWN)

		return codes

	def add(self, texts):
		if isinstance(texts, list):
			for text in texts:
				self.add_one(text)
		else:
			self.add_one(texts)	

	def add_one(self, text):
		tokens = self.tokenize(text)
		for t in tokens:
			if self._tf.has_key(t):
				self._tf[t] += 1
			else:
				self._tf[t] = 1

	
	def create_code(self):
		self._code = {}
		tf = sorted(self._tf.items(), key = lambda k: -k[1])
		
		c = 0
		for token, freq in tf:
			self._code[token] = c
			c += 1
		
		self._CODE_UNKNOWN = c
		self._n_code = c + 1 # one more code for the unknown character
		self._tf = {}

	@classmethod
	def tokenize(self, text):
		text = text.decode('utf8').lower()
		tokens = []
		
		buf = ''
		for t in text:
			if t >= 'a' and t <= 'z':
				buf += t
				continue

			if not buf == '':
				tokens.append(buf)
				buf = ''
			
			if not t == ' ':
				tokens.append(t)

		return tokens

def init(sample = None):
	import blogger
	from const import N_EMO, DIR_TEXT, PKL_TFCODER
	from utils import progbar
	
	n_emo = N_EMO

	coder = TfCoder()
	pbar = progbar.start(n_emo)

	for eid in range(n_emo):
		lines =  open(DIR_TEXT + '%d.txt'%(eid), 'r').read().split('\n')
		coder.add(lines)
		pbar.update(eid + 1)

	pbar.finish()

	coder.create_code()

	codesA = coder.code(sample)
	cPickle.dump(coder, open(PKL_TFCODER, 'w'))

	coder = cPickle.load(open(PKL_TFCODER, 'r'))
	codesB = coder.code(sample)

	eflag = False
	if not len(codesA) == len(codesB):
		print 'Error: length mismatch'
		eflag = True
	else:
		for i in range(len(codesA)):
			if not codesA[i] == codesB[i]:
				print 'Error: different code at the position %d'%(i)
				eflag = True
	
	if not eflag:
		print 'Info: coder created at %s'%(PKL_TFCODER)
		print 'Info: n_code = %d'%(coder.n_code())

def default_load():
	from const import PKL_TFCODER
	return cPickle.load(open(PKL_TFCODER, 'r'))

if __name__ == '__main__':
	text = u'哗一下又涨几十个粉1880个粉丝最少有1500是和他的姘头输送的，剩下380个是我认识的我自己根本一个粉丝也没有，唯一一个铁杆粉丝还被我做通思想工作，傲娇地离开了'
	tokens = TfCoder.tokenize(text)
	print '/'.join(tokens)

	#init(text)
