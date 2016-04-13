#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.01.24
Modified: 2016.02.23
Description: a tool used to process text content of microblogs in Weibo
'''

import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from share import emotica

################################### CONST ########################################

pattern_kr = re.compile(u'[\u1100-\u11ff\u2e80-\u33ff\uac00-\ud7af\u3130–\u318f\u3200–\u32ff\ua960–\ua97f\uac00-\ud7a3\ud7b0–\ud7ff\uff00–\uffef]+')

pattern_zh = re.compile(u'[\u4e00-\u9fa5]')

################################### PRIVATE #########################################

def extract_main(blog):
	'''
	extract the part of text created by the author
	'''
	text = remove_forward(blog)
	text = remove_share(text)
	text = remove_mention(text)
	text = remove_space(text)	

	return text

def remove_space(text):
	'''
	delete the space at the front and the end
	turn spaces in the middle of the text to only one space
	'''

	text = text.strip()
	text = re.sub('\s+', ' ', text)

	return text

def remove_korean(text):
	'''
	delete all the korean characters
	'''
	return re.sub(pattern_kr, ' ', text)

def remove_forward(blog):
	return re.sub('//@.+$', '', blog, re.DOTALL)

def remove_share(blog):
	text = re.sub(u'【[^】]+】.*http://t.cn/\w+', '', blog, re.DOTALL)
	text = re.sub(u'(\s+\S+:)?http://t.cn/\w+\s*$', '', text)
	text = re.sub(u'^.*http://t.cn/\w+\s+（\S+自 @\S+）', '', text, re.DOTALL)

	return text

def remove_mention(blog):
	return re.sub('@\S+', '', blog)

def contain_zh(text):
	'''
	check where the text is composed of at least one chinese character
	'''
	m = re.search(pattern_zh, text)
	return not m == None

################################### PUBLIC #########################################

def is_valid(blog, check_emo = True):
	'''
	shortcut for checking where the blog is valid 
	'''
	return extract(blog, check_emo) is not None

def extract(blog, check_emo = True):
	'''
	analyse the blog, if it contains only one distinct emoticon and the text part is not empty,
	return the text part and emoticon, otherwise None is returned.
	'''

	# extract the part created by user

	blog = blog.decode('utf8')
	main_blog = extract_main(blog)

	# extract and check emoticons

	emoticons = emotica.extract_emoticons(main_blog)
	

	if check_emo:
		if len(emoticons) == 0:
			#print 'No Emoticons'
			return None
		else:
			phrases = [emotica.remove_prefix(emo) for emo in emoticons]
			phrases = set(phrases)
			if len(phrases) > 1:
				'''
				more than one distinct emoticons within this blog
				'''
				#print 'Too many phrases'
				return None

			emo = emotica.remove_prefix(list(phrases)[0])
	else:
		emo = None
	# extract and check the text

	text = emotica.remove_emoticons(main_blog)
	text = remove_korean(text)
	text = remove_space(text)

	if len(text) == 0:
		'''
		no text content
		'''
		#print 'No text'
		return None

	if not contain_zh(text):
		'''
		no chinese character
		'''	
		return None

	text = re.sub('\s+', ' ', text).strip()
	return text, emo

def simple_extract(blog, check_emo = True):
	'''
	analyse the blog, if it contains emoticon(s) and the text part is not empty,
	return the text part and emoticons, otherwise None is returned.
	'''

	# extract the part created by user

	blog = blog.decode('utf8')
	main_blog = extract_main(blog)

	# extract and check emoticons

	emoticons = emotica.extract_emoticons(main_blog)
	

	if check_emo:
		if len(emoticons) == 0:
			#print 'No Emoticons'
			return None
		else:
			phrases = [emotica.remove_prefix(emo) for emo in emoticons]
			phrases = set(phrases)
			emos = [emotica.remove_prefix(phrase) for phrase in phrases]
	else:
		emos = []
	# extract and check the text

	text = emotica.remove_emoticons(main_blog)
	text = remove_korean(text)
	text = remove_space(text)

	if len(text) == 0:
		'''
		no text content
		'''
		#print 'No text'
		return None

	if not contain_zh(text):
		'''
		no chinese character
		'''	
		return None

	text = re.sub('\s+', ' ', text).strip()
	return text, emos

def decompose(blog):
	'''
	decompose the blogs into text and list of emos
	'''

	# extract the part created by user

	blog = blog.decode('utf8')
	main_blog = extract_main(blog)

	# extract and check emoticons
	emoticons = emotica.extract_emoticons(main_blog)
	
	if not len(emoticons) == 0:
		phrases = [emotica.remove_prefix(emo) for emo in emoticons]
		phrases = set(phrases)
		emos = [emotica.remove_prefix(phrase) for phrase in phrases]
	else:
		emos = []

	# extract and check the text
	text = emotica.remove_emoticons(main_blog)
	text = remove_korean(text)
	text = remove_space(text)
	text = re.sub('\s+', ' ', text).strip()

	return text, emos


if __name__ == '__main__':
	blog = u'【重庆回应长江水变红：未测出有毒物质】核心提示：9月7日，重庆环保局回应长江变“红河”称，河水中未检测出有毒有害物质，河水变红系水中铁离子含量过高所致。... http://t.cn/zWDrb4x  [哈哈]富含铁离子，还富含各种矿物质和维生素，这不是高级纯天然矿泉水吗，环保菊长先来一口？'
	print extract_main(blog)

	res = extract(blog)
	if res == None:
		print 'INVALID'
	else:
		text, emo = res
		print '%s|%s'%(text, emo)
		print [text, ]

