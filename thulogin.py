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

import md5
from optparse import OptionParser

class UrlOpener:
	__DEFAULT_TIMEOUT = 10

	def __init__(self):
		self.timeout = self.__DEFAULT_TIMEOUT
		self.header = UrlOpener.default_header()
		self.opener = UrlOpener.build_opener()

	@classmethod
	def default_header(self):
		return {
			'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:38.0) Gecko/20100101 Firefox/38.0',
			'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
			'Accept-Encoding':'gzip, deflate', 
			'Accept-Language':'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
			'Connection':'keep-alive',
			'Referer':'http://net.tsinghua.edu.cn/wired/',
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

def to_md5(src):
	m1 = md5.new()
	m1.update(src)
	return m1.hexdigest()

def url_login(ac, pw):
	url = 'http://net.tsinghua.edu.cn/do_login.php'

	params = {
		'action':'login',
		'username':ac,
		'password':'{MD5_HEX}' + to_md5(pw),
		'ac_id':1,
	}

	param = urllib.urlencode(params)
	return '%s?%s'%(url, param)

def url_logout():
	url = 'http://net.tsinghua.edu.cn/do_login.php'

	params = {
		'action':'logout',
	}

	param = urllib.urlencode(params)
	return '%s?%s'%(url, param)

opener = UrlOpener()
opener.set_cookie('tunet=idu%0AspeechTHU2015; thuwebcookie=2043441062.20480.0000')

def login(ac, pw):
	url = url_login(ac, pw)
	content = opener.urlopen(url)
	print content

def logout():
	url = url_logout()
	content = opener.urlopen(url)
	print content


def main():
	optparser = OptionParser()
	optparser.add_option('-u', '--user', action='store', type='string', dest='user')
	optparser.add_option('-p', '--password', action='store', type='string', dest='password')
	optparser.add_option('-o', '--logout', action='store_true', dest='logout')

	opts, args = optparser.parse_args()
	
	if opts.logout:
		logout()
	else:
		login(opts.user, opts.password)

if __name__ == '__main__':
	main()
	
