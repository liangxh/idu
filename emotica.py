#! /usr/bin/env python
# -*- coding: utf-8 -*- 
'''
Author: Xihao Liang
Created: 2016.01.24
Modified: 2016.01.25
Description: a tool used to analyse emoticons on Weibo
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import re
import json
from const import JSON_EMO

class Emotica:
	def __init__(self, fname_emo = JSON_EMO):
		'''
		initization of the data of emoticons
		'''
		self.load(fname_emo)
		self.init_re_pattern()
		self.init_remove_prefix()

	def load(self, fname):
		'''
		read data of emoticons from JSON_EMO, which is saved from URL_EMO
		'''

		js = json.load(open(fname, 'r'))['data']

		self._collections = {}
		emo_pairs = []
		emo_pairs.append(('usual', js['usual']['norm']))
		#emo_pairs.extend(js['brand']['norm'].items()) # covered in js['more']
		emo_pairs.extend(js['more'].items())

		for emo_name, emo_list in emo_pairs:
			if self._collections.has_key(emo_name):
				print 'CRASH: %s'%(emo_name)

			self._collections[emo_name] = [{'phrase': emo['phrase'][1:-1], 'url': emo['url']} for emo in emo_list]

	def init_re_pattern(self):
		all_phrases = []
		for emo_list in self._collections.values():
			for emo in emo_list:
				all_phrases.append(emo['phrase'])
	
		self._pattern_emo = re.compile(r'\[((?:%s))\]'%(')|(?:'.join(all_phrases)))

	def init_remove_prefix(self):
		self._remove_prefix = {}
		for emo_name, emo_list in self._collections.items():
			phrases = [emo['phrase'] for emo in emo_list]
			if self.have_eng_prefix(phrases):
				for phrase in phrases:
					mp = self.remove_eng_prefix(phrase)
					if not mp == None:
						self._remove_prefix[phrase] = mp
			else:
				prefix_len = self.get_prefix_length(phrases)
				if prefix_len > 0:
					for phrase in phrases:
						self._remove_prefix[phrase] = phrase[prefix_len:]


	def remove_prefix(self, phrase):
		if self._remove_prefix.has_key(phrase):
			return self._remove_prefix[phrase]
		else:
			return phrase

	def has_emoticon(self, text):
		m = re.search(self._pattern_emo, text)

		#if m:
		#	print m.group(1)

		return not m == None

	def remove_emoticons(self, text):
		return re.sub(self._pattern_emo, ' ', text)

	def extract_emoticons(self, text):
		return re.findall(self._pattern_emo, text)

	def have_eng_prefix(self, phrases):
		for phrase in phrases[0:3]:
			p = self.remove_eng_prefix(phrase)
			if not p == None:
				return True

		return False

	def get_prefix_length(self, phrases):
		if len(phrases) <= 1:
			return None

		prefix = phrases[0]
		match = 0
		miss = 0
		threshold = 2
		for phrase in phrases[1:]:
			if phrase.startswith(prefix):
				continue

			hit = False
			for i in range(1, len(prefix)):
				p = prefix[0:-i]
				if phrase.startswith(p):
					prefix = p
					hit = True
					break

			if not hit:
				miss += 1
				if miss > threshold:
					return 0

		return len(prefix)

	def remove_eng_prefix(self, phrase):
		'''
		only pattern like #lxhx开心 -> 开心 is supported
		'''
		m = re.match(u'^\w+([\u4e00-\u9fa5]+)\w*$', phrase) 
		if m:
			return m.group(1)

		return None
	
	def get(self, key):
		'''
		return the collection of $key
		'''
		if self._collections.has_key(key):
			return self._collections[key]

		return None

	def keys(self):
		return self._collections.keys()

	def values(self):
		return self._collections.values()

	def items(self):
		return self._collections.items()

if __name__ == '__main__':
	from utils import zhprocessor

	db = Emotica()

	print '# of Emoticon Collections: ',  len(db.keys())
	print '# of Emoticons: ', sum([len(v) for v in db.values()])

	for v in db.values():
		break
		for vi in v:
			print '%s/%s'%(vi['phrase'], vi['value'])

	t = u'你好[泪] [泪] 我是 [泪] 梁錫豪'
	t = zhprocessor.simplify(t)
	print t
	print db.has_emoticon(t)

