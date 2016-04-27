#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.26
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re
import cPickle

from share import emotica

def main():
	N_EMO = 90
	phrases = open('data/eid.txt', 'r').read().decode('utf8').split('\n')[:N_EMO]
	
	eid = {}
	for i, phrase in enumerate(phrases):
		eid[phrase] = i

	urls = [[] for i in range(N_EMO)]

	for emolist in emotica.values():
		for emodata in emolist:
			phrase = emotica.remove_prefix(emodata['phrase'])
			if eid.has_key(phrase):
				urls[eid[phrase]].append(emodata['url'])

	html = ''
	for i, phrase in enumerate(phrases):
		html += '%d. %s:<br>\n'%(i, phrase) + ''.join(['.....%d<img src="%s"></img><br>\n'%(j, url)  for j, url in enumerate(urls[i])])
	open('output/top90emo_raw.html', 'w').write(html.encode('gbk'))

def extract():
	N_EMO = 90
	phrases = open('data/eid.txt', 'r').read().decode('utf8').split('\n')[:N_EMO]

	content = open('output/top90emo.html', 'r').read()
	urls = re.findall('"([^"]+)"', content)
	
	cPickle.dump((phrases, urls), open('data/emourls.pkl', 'w'))
	html = ''
	for i in range(N_EMO):
		phrase = phrases[i]
		url = urls[i]
		html += '%d. %s: <img src="%s"><br>\n'%(i, phrase, url)

	open('data/emodata/emourl.html', 'w').write(html.encode('gbk'))
	

def load():
	N_EMO = 90
	phrases = open('data/eid.txt', 'r').read().decode('utf8').split('\n')[:N_EMO]

	content = open('output/top90emo.html', 'r').read()
	urls = re.findall('"([^"]+)"', content)
	return phrases, urls

def build_board(title, items, sample):
	phrases, urls = load()
	
	html = '<h3>%s</h3>\n'%(title)
	html += '<hr>'
	html += '<div style="font-size:12px">Sample: %s</div>'%(sample)
	html += '<hr>'
	html += '<table style="font-size:12px; border-collapse:collapse;">\n'

	l = 0
	for p, u in zip(phrases, urls):
		l += 1

		html +='<tr style="border-bottom:#0000FF solid 1px;">'
		html += '<td style="padding-bottom:10px">%d. <img src="%s"> [%s]</td>'%(l, u, p)

		#html += '<td>'
		#for i in range(5):
		#	html += '<input type="checkbox">'
		#html += '</td>'		
		for item in items:
			html += '<td style="text-align:center; padding-left:20px; padding-right:20px">%s</td>'%(item)
		

		html += '</tr>\n'
	html += '</table>'
	html = html.encode('gbk')

	open('data/emodata/q.html', 'w').write(html)


if __name__ == '__main__':
	#main()
	extract()
	#title = u'Emoji愉悦度 (情感状态的正负特性)'; items = '非常开心 开心 中等 不开心 非常不开心'.split(' '); sample = ''
	#title = u'Emoji激活度 (个体的神经生理激活水平)'; items = u'强 稍强 中等 稍弱 弱'.split(' '); sample = u'大笑 > 微笑 > 沉默'
	#title = u'Emoji优势度 (个体对情景和他人的控制状态)'; items = u'主动 稍主动 中等 稍被动 被动'.split(' '); sample = u'愤怒 > 恐惧'

	#build_board(title, items, sample)
