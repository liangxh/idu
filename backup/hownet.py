#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2015.07
Description: lib for using HowNet
'''

import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

FOLDER_HOWNET = 'data/HowNet/'

MARK = {'#', '%', '$', '*', '&', '~', '@', '?', '^', '!'}

class Word:
	def __init__(self, line):
		self.raw = line
		l = line.replace('\r', '').replace('\n', '').replace('(', '').replace(')', '')
		l = re.sub(r'\s+', ' ', l)
		items = l.split(' ')
		for i in range(3, len(items)):
			items[2] += items[i]

		self.token = items[0]
		self.pos = items[1]
		self.parse_attri(items[2].replace(' ', ''))
	
	def parse_attri(self, attri_str):
		attri1 = ''
		attri2 = []
		attri3 = {}
		attri4 = {}
		notional = True

		attris = attri_str.split(',')
		if len(attris) == 1:
			attri = attris[0]
			if attri.startswith('{'):
				notional = False
				attri1 = attri[1:len(attri) - 1]
			else:
				attri1 = attri
		else:
			for attri in attris:
				if attri[0] in MARK:
					mark = attri[0]
					attri = attri[1:len(attri)]
					if attri4.has_key(mark):
						attri4[mark].add(attri)
					else:
						attri4[mark] = {attri, }
				elif not attri.find('=') == -1:
					items = attri.split('=')
					attri3[items[0]] = items[1]
				else:
					attri2.append(attri)

			if len(attri2) > 0:
				attri1 = attri2.pop(0)

		self.attri1 = attri1
		self.attri2 = attri2
		self.attri3 = attri3
		self.attri4 = attri4
		self.notional = notional

class HowNet:
	def __init__(self,
		folder = FOLDER_HOWNET,
		alpha = 1.6,
		beta = [0.5, 0.2, 0.17, 0.13],
		gamma = 0.2,
		delta = 0.2, 
	):
		if not folder.endswith('/'):
			folder += '/'
		self.fname_whole = folder + 'wordsimilarity/whole.dat'
		self.fname_glossary = folder + 'wordsimilarity/glossary.dat'
		self.fname_emo_pos = folder + 'sentiment/emo_pos_chi.txt'
		self.fname_emo_neg = folder + 'sentiment/emo_neg_chi.txt'

		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.delta = delta

		self.read_whole(self.fname_whole)
		self.read_glossary(self.fname_glossary)
		self.emo_neg = None
		self.emo_pos = None

	def get_emo_pos(self):
		if not self.emo_pos:
			self.emo_pos = self.read_emo(self.fname_emo_pos)
		return self.emo_pos

	def get_emo_neg(self):
		if not self.emo_neg:
			self.emo_neg = self.read_emo(self.fname_emo_neg)
		return self.emo_neg

	def read_emo(self, fname):
		fobj = open(fname, 'r')
		lines = fobj.readlines()
		fobj.close()

		emos = []		
		for i in range(2, len(lines)):
			line = lines[i].decode('utf8')
			line = re.sub(r'\s*', '', line)
			emos.append(line)

	def read_whole(self, fname):
		fobj = open(fname, 'r')
		lines = fobj.readlines()
		fobj.close()

		tran = {}
		parent = {}

		for line in lines:
			l = line.decode('utf-8')
			l = l.replace('\r\n', '')
			l = re.sub(r'\s+', ' ', l)
			items = l[1:len(l)].split(' ')
			tran[items[1]] = int(items[0])
			parent[int(items[0])] = int(items[2])

		self.tran = tran
		self.parent = parent

	def read_glossary(self, fname):	
		fobj = open(fname, 'r')
		lines = fobj.readlines()
		fobj.close()

		self.words = {}
		for line in lines:
			l = line.decode('utf-8')
			word = Word(l)
			if self.words.has_key(word.token):
				self.words[word.token].append(word)
			else:
				self.words[word.token] = [word, ]

	def sim_word(self, word1, word2):
		s1 = self.sim1(word1, word2)
		s2 = self.sim2(word1, word2)
		s3 = self.sim3(word1, word2)
		s4 = self.sim4(word1, word2)
		sims = [s1, s2, s3, s4]
		
		s = 0.
		tmp = 1.
		for i in range(4):
			tmp *= sims[i]
			s += tmp * self.beta[i]
		#print '--sim result--'
		#print sims
		#print s
		return s

	def sim_p(self, p1, p2):
		flag1 = not self.tran.has_key(p1)
		flag2 = not self.tran.has_key(p2)

		if flag1 and flag2:
			if flag1 == flag2:
				return 1.
			else:
				return 0.
		elif flag1 or flag2:
			return self.gamma

		pid1 = self.tran[p1]
		pid2 = self.tran[p2]
		
		path1 = [pid1, ]
		cid = pid1
		nid = self.parent[cid]
		while (not cid == nid):
			path1.append(nid)
			cid = nid
			nid = self.parent[cid]
		path1.reverse()

		path2 = [pid2, ]
		cid = pid2
		nid = self.parent[cid]
		while (not cid == nid):
			path2.append(nid)
			cid = nid
			nid = self.parent[cid]
		path2.reverse()

		loop = min(len(path1), len(path2))
		i = 0
		while i < loop:
			if not path1[i] == path2[i]:
				break
			i += 1
		
		#print '--sim_p--'
		#print p1, path1
		#print p2, path2

		if i == 0:
			d = len(path1) + len(path2) - 1
		else:
			d = len(path1) + len(path2) - 2 * i
		#print d
		s1 = self.alpha / (self.alpha + d)
		#print s1
		return s1

	def sim_set(self, set1, set2):
		if len(set1) + len(set2) == 0:
			return 1

		if len(set1) > len(set2):
			set1, set2 = (set2, set1)
		N = len(set1)

		sims = []
		for p1 in set1:
			for p2 in set2:
				sims.append((self.sim_p(p1, p2), p1, p2))
		sims = sorted(sims, key = lambda k:k[0], reverse = True)

		count = 0
		del_p = set()
		s_sum = 0.
		for s, p1, p2 in sims:
			if p1 in del_p or p2 in del_p:
				continue
			#print p1, p2, s
			del_p.add(p1)
			del_p.add(p2)
			s_sum += s
			count += 1
			if count == N:
				break
		s_sum += (len(set2) - len(set1)) * self.delta
		return s_sum / len(set2)
		

	def sim1(self, word1, word2):
		return self.sim_p(word1.attri1, word2.attri1)

	def sim2(self, word1, word2):
		return self.sim_set(word1.attri2, word2.attri2)

	def sim3(self, word1, word2):
		keys1 = word1.attri3.keys()
		keys2 = word2.attri3.keys()
		keys = set(keys1) | set(keys2)
		if len(keys) == 0:
			return 1

		s_sum = 0.
		for k in keys:
			if (not k in keys1) or (not k in keys2):
				continue
			s_sum += self.sim_p(word1.attri3[k], word2.attri3[k])
		return s_sum / len(keys)

	def sim4(self, word1, word2):
		keys1 = word1.attri4.keys()
		keys2 = word2.attri4.keys()
		keys = set(keys1) | set(keys2)
		if len(keys) == 0:
			return 1

		s_sum = 0.
		for k in keys:
			if (not k in keys1) or (not k in keys2):
				s_sum += self.delta
			else:
				s_sum += self.sim_set(word1.attri4[k], word2.attri4[k])
		return s_sum / len(keys)

	############################### PUBLIC ##############################################
	
	def sim(self, token1, token2):
		if not (self.words.has_key(token1)):
			#print 'Warning: %s is not in the dictionary'%(token1)
			return 0
		if not (self.words.has_key(token2)):
			#print 'Warning: %s is not in the dictionary'%(token2)
			return 0

		sim = -1
		for word1 in self.words[token1]:
			for word2 in self.words[token2]:
				s = self.sim_word(word1, word2)
				if s > sim:
					sim = s
		return sim

	def support(self, token):
		return self.words.has_key(token)

def test_sim(text):
	text = text.split(' ')
	token1 = text[0]
	token2 = text[1]
	print hownet.sim(token1, token2)
#test_sim(u'工人 教师')

def main():
	hownet = HowNet(FOLDER_HOWNET)

if __name__ == '__main__':
	main()
