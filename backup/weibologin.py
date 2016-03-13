#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (c) 2013 Qin Xuye <qin@qinxuye.me>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on 2013-6-8

@author: Chine
'''

import rsa
import base64
import binascii
import re
import json

import urllib
import urllib2
import socket


import cookielib


TIMEOUT = 10

URL_WRONG_PW = "http://weibo.com/ajaxlogin.php?framelogin=1&amp%3Bcallback=parent.sinaSSOController.feedBackUrlCallBack&retcode=101&reason=%B5%C7%C2%BC%C3%FB%BB%F2%C3%DC%C2%EB%B4%ED%CE%F3"

import StringIO, gzip

def build_opener():	
	cj = cookielib.CookieJar()

	cookie_processor = urllib2.HTTPCookieProcessor(cj)
	null_proxy_handler = urllib2.ProxyHandler({})

	urlopener = urllib2.build_opener(cookie_processor, null_proxy_handler)

	return urlopener


def ungzip(data):
	'''
	decompress the response package
	'''
	compressedstream = StringIO.StringIO(data)  
	gziper = gzip.GzipFile(fileobj = compressedstream)
	content = gziper.read()
	return content

'''
def show_cookie():
	for ck in cj:
		print ck.name, ':', ck.value
'''

headers = {
		'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
		'Accept-Encoding':'gzip, deflate, sdch', 
		'Accept-Language':'zh-TW,zh;q=0.8,en-US;q=0.6,en;q=0.4',
		'Connection':'keep-alive',
		'Cookie':'',
	}

def urlopen(urlopener, url, data = None):
	'''
	send request to URL with header $request_header
	'''
	req = urllib2.Request(url, headers = headers)

	try:
		resp = urlopener.open(req, data = data, timeout = TIMEOUT)
		content = resp.read()

		is_gzip = resp.headers.dict.get('content-encoding') == 'gzip'
		if is_gzip:
			content = ungzip(content)

		return content
	except urllib2.URLError, e:
		print e.reason
		return None
	except urllib2.HTTPError, e:
		print '[ERRNO %d] %s'%(e.code, e.reason)
		return None
	except socket.timeout, e:
		print '[Timeout] %s'%(e.reason)
		return None




class WeiboLogin(object):
	def __init__(self, username, passwd, urlopener = None):
		self.username = username
		self.passwd = passwd
		
		if urlopener is None:
			self.urlopener = build_opener()
		else:
			self.urlopener = urlopener

	def get_user(self, username):
		username = urllib.quote(username)
		return base64.encodestring(username)[:-1]
	
	def get_passwd(self, passwd, pubkey, servertime, nonce):
		key = rsa.PublicKey(int(pubkey, 16), int('10001', 16))
		message = str(servertime) + '\t' + str(nonce) + '\n' + str(passwd)
		passwd = rsa.encrypt(message, key)
		return binascii.b2a_hex(passwd)
	
	def prelogin(self):
		username = self.get_user(self.username)
		prelogin_url = 'http://login.sina.com.cn/sso/prelogin.php?entry=sso&callback=sinaSSOController.preloginCallBack&su=%s&rsakt=mod&client=ssologin.js(v1.4.5)' % username
		data = urlopen(self.urlopener, prelogin_url)
		regex = re.compile('\((.*)\)')

		json_data = regex.search(data).group(1)
		data = json.loads(json_data)
			
		return str(data['servertime']), data['nonce'], data['pubkey'], data['rsakv']
		
	def login(self):
		login_url = 'http://login.sina.com.cn/sso/login.php?client=ssologin.js(v1.4.5)'
		
		res = self.prelogin()
		if res == None:
			print '[warning] prelogin failed'
			return None

		servertime, nonce, pubkey, rsakv = res
		postdata = {
			'entry': 'weibo',
			'gateway': '1',
			'from': '',
			'savestate': '7',
			'userticket': '1',
			'ssosimplelogin': '1',
			'vsnf': '1',
			'vsnval': '',
			'su': self.get_user(self.username),
			'service': 'miniblog',
			'servertime': servertime,
			'nonce': nonce,
			'pwencode': 'rsa2',
			'sp': self.get_passwd(self.passwd, pubkey, servertime, nonce),
			'encoding': 'UTF-8',
			'prelt': '115',
			'rsakv' : rsakv,
			'url': 'http://weibo.com/ajaxlogin.php?framelogin=1&amp;callback=parent.sinaSSOController.feedBackUrlCallBack',
			'returntype': 'META'
		}
		postdata = urllib.urlencode(postdata)
		text = urlopen(self.urlopener, login_url, postdata)

		# Fix for new login changed since about 2014-3-28
		ajax_url_regex = re.compile('location\.replace\("(.*)"\)')
		matches = ajax_url_regex.search(text)
		if matches is not None:
			ajax_url = matches.group(1)

			m = re.search(r'reason=([^&]+)', ajax_url)
			if m is not None:				
				print urllib.unquote(m.group(1)).decode('gbk')
				return False

			text = urlopen(self.urlopener, ajax_url)

		print text
		regex = re.compile('\((.*)\)')
		json_data = json.loads(regex.search(text).group(1))
		print json_data

		result = json_data['result'] == True
		if result is False and 'reason' in json_data:
			print '[warning] ', json_data['reason']

		return result

if __name__ == '__main__':
	loginer = WeiboLogin('danhou13@sina.cn', 'devil1024')
	res = loginer.login()
	#show_cookie()
 

