#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.01
Description: A wrapper for urllib, cookie may be added later
'''

import urllib
import urllib2
import socket
import cookielib

TIMEOUT = 10
import StringIO, gzip

class UrlOpener:
	__DEFAULT_TIMEOUT = 10

	def __init__(self):
		self.timeout = self.__DEFAULT_TIMEOUT
		self.header = UrlOpener.default_header()
		self.opener = UrlOpener.build_opener()

	@classmethod
	def default_header(self):
		return {
			'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
			'Accept-Encoding':'gzip, deflate, sdch', 
			'Accept-Language':'zh-TW,zh;q=0.8,en-US;q=0.6,en;q=0.4',
			'Connection':'keep-alive',
			'Cookie':'',
			}

	@classmethod
	def build_opener(self):
		null_proxy_handler = urllib2.ProxyHandler({})
		opener = urllib2.build_opener(null_proxy_handler)

		return opener

	@classmethod
	def ungzip(self, data):
		'''
		decompress the response package
		'''
		compressedstream = StringIO.StringIO(data)  
		gziper = gzip.GzipFile(fileobj = compressedstream)    
		content = gziper.read()

		return content

	def set_timeout(self, timeout):
		self.timeout = timeout

	def set_cookie(self, cookie):
		self.header['Cookie'] = cookie

	def clear_cookie(self):
		self.header['Cookie'] =  ''

	def empty_cookie(self):
		return self.header['Cookie'] == ''

	def urlopen(self, url, data = None):
		req = urllib2.Request(url, headers = self.header)

		try:
			resp = self.opener.open(req, data = data, timeout = self.timeout)
			content = resp.read()

			is_gzip = resp.headers.dict.get('content-encoding') == 'gzip'
			if is_gzip:
				content = UrlOpener.ungzip(content)

			return content
		except urllib2.URLError, e:
			print e.reason
			return None
		except urllib2.HTTPError, e:
			print '[ERRNO %d] %s'%(e.code, e.reason)
			return None
		except socket.timeout, e:
			print '[Timeout] ', e
			return None

def test():
	opener = UrlOpener()
	content = opener.urlopen('http://www.baidu.com/')
	print content

if __name__ == '__main__':
	test()
	
