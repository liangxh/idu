#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.01.25
Modified: 2016.01.25
Description: functions for parsing comments in Weibo
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import re
import time
import json
import traceback

import urllib
import urllib2
import socket

from bs4 import BeautifulSoup

RETCODE = 6102

def url_comment(uid, mid):  #, myid
	'''
	return the url for a small comment list
	'''

	url = 'http://weibo.com/aj/v6/comment/small'

	params = {
		'ajwvr':6,
		'act':'list',
		'mid':mid,
		'uid':uid,    # myid,
		'isMain':True,
		'dissDataFromFeed':'[object Object]',
		'ouid':uid,
		'location':'page_100505_home', #???
		'comment_type':1,
		'_t':0,
		'__rnd':0,	#1453814722537
		'retcode':RETCODE,
	}

	param = urllib.urlencode(params)
	return '%s?%s'%(url, param)

def url_comment_page(page, comm_id, max_id):
	'''
	return the url for a comment page
	'''

	url = 'http://weibo.com/aj/v6/comment/big'

	params = {
		'ajwvr':6,
		'id':comm_id,
		'max_id':max_id,
		'page':page,
		'__rnd':0, #1453883315616
		'retcode':RETCODE,
	}

	param = urllib.urlencode(params)
	return '%s?%s'%(url, param)

def soup2comments(soup):
	'''
	private function called by $get_comments
	extract comments from soup of the response 
	'''

	comment_list = soup.find('div', attrs={'class':'list_ul'})

	divs_comment = comment_list.findAll('div', attrs={'comment_id':re.compile(r'\d+')})
	comments = []

	for div_comment in divs_comment:
		wb_text = div_comment.find('div', attrs={'class':'WB_text'})

		m = re.match(u'^\s*([^：]+)：(.*)$', wb_text.text, re.DOTALL)
		from_name = m.group(1)
		text = m.group(2).replace('：', ':')
		to_name = None

		m = re.match(u'回复:?@([^:]+):(.*)\s+', text, re.DOTALL)
		if m:
			to_name = m.group(1).strip()
			text = m.group(2).strip()

		comments.append({'from_name':from_name, 
							'to_name':to_name,
							'text':text})

	comments.reverse()

	return comments

def soup2ids(soup):
	'''
	private function called by $get_comments
	at ids from the usercards, not necessary so far
	'''

	uids = {}
	usercards = soup.findAll('a', attrs={'usercard':re.compile('.*id=\d+.*')})

	for usercard in usercards:
		uname = usercard.text.decode('utf8')
		if uids.has_key(uname):
			continue

		card = usercard.get('usercard')
		m = re.search('id=(\d+)', card)
		uid = m.group(1)
		uids[uname] = uid

	return uids

def add_emoticons_text(html):
	'''
	add title after <img> for each emoticon
	otherwise, soup.text will skip the emoticons
	'''
	return re.sub('<img[^>]*title="(?P<title>[^"]+)"[^>]*type="face"[^>]*>', '\g<title>', html)

def get(urlopener, uid, mid, show_result = False, show_max = 10):
	url = url_comment(uid, mid)
	response = urlopener.urlopen(url)

	if response == None:
		print 'none response'
		return

	try:
		res = json.loads(response)
	except ValueError:
		m = re.search('location.replace\("([^"]+)"\)', response)
		if m == None:
			print 'location.replace expected but not found'
			return None
		else:
			url = m.group(1)
			print url
			return None

	comments_count = res['data']['count']

	html = res['data']['html']
	html = add_emoticons_text(html)
	
	soup = BeautifulSoup(html, 'html.parser')

	############ Parse Comments ############
	comments = []
	ids = {}

	cardmore = soup.find('a', attrs={'class':re.compile('.*WB_cardmore.*')})
	if not cardmore:
		'''
		all comments contained in this response
		'''
		ids = soup2ids(soup)
		comments = soup2comments(soup)

	else:
		'''
		comments are shown in severals pages
		more requests for pages required
		'''
		
		comm_id = mid

		first_comment = soup.find('div', attrs={'comment_id':re.compile('\d+')})
		max_id = first_comment.get('comment_id')

		page = 0
		while (len(comments) < comments_count):
			#print page, len(comments)
			page += 1
			url = url_comment_page(page, comm_id, max_id)
			response = urlopener.urlopen(url)
			
			if response == None:
				'''
				unexpected error while parsing pages
				'''
				return None

			ret = json.loads(response)

			html = ret['data']['html']
			html = add_emoticons_text(html)
			
			soup = BeautifulSoup(html, 'html.parser')
			comm = soup2comments(soup)

			if len(comm) > 0:
				comments.extend(comm)
				ids.update(soup2ids(soup))
			else:
				'''
				comments_count may be wrong as advertisement is omitted by Sina
				just ignore it
				'''
				break

	if show_result:
		for i, c in enumerate(comments):
			id2 = []
			for name in [c['from_name'], c['to_name']]:
				if name == None:
					id2.append('-1')
				elif not ids.has_key(name):
					id2.append('0')
				else:
					id2.append(ids[name])

			if i < show_max:
				if c['to_name'] == None:
					print '%s: %s'%(c['from_name'], c['text'])
				else:
					print '%s: RE@%s : %s'%(c['from_name'], c['to_name'], c['text'])
			elif i == show_max:
				print '...(%d in total)'%(len(comments))

	return comments, ids
	

if __name__ == '__main__':
	'''
	# see weiboparser.test_one

	#uid = '1427605041'
	#mid = '3523152740977956'
	
	uid, mid = ('1427605041', '3509858479270286')
	#uid, mid = ('1448253167', '3493392086202628')
	#uid, mid = ('1694167544', '3506425500263895')

	from utils.urlopener import UrlOpener
	from const import WEIBO_COOKIE, WEIBO_MYID

	urlopener = UrlOpener()
	urlopener.set_cookie(WEIBO_COOKIE)

	ret = get(urlopener, uid, mid, WEIBO_MYID, show_result = True)
	if ret == None:
		print 'failed'

	#comment_page(1)
	'''
	pass
	
