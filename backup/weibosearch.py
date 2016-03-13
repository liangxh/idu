'''
extracted from weiboparser.py
'''

def url_search(uid, start_time, key_word):
	'''
	return the url for searching blog, not used 
	'''

	url = 'http://weibo.com/p/100505%s/home'%(uid)

	start_timestamp = time.mktime(time.strptime(start_time, '%Y-%m-%d'))
	end_time = time.strftime('%Y-%m-%d', time.localtime(start_timestamp + 86400))

	params = {
		'pids':'Pl_Official_MyProfileFeed__24',
		'is_ori':1,
		'is_forward':1,
		'is_text':1,
		'is_pic':1,
		'is_video':1,
		'is_music':1,
		'key_word':key_word,
		'start_time':start_time,
		'end_time':end_time,
		'is_search':1,
		'is_searchadv':1,
		'ajaxpagelet':1,
		'ajaxpagelet_v6':1,
		'__ref':'/p/100505%s/home'%(uid),
		'_t':'FM_145372470006591',
	}

	param = urllib.urlencode(params)
	return '%s?%s'%(url, param)

def search(uid, start_time, key_word):
	url = url_search(uid, start_time, key_word)
	response = request(url)

	if response == None:
		return

	m = re.search(r'<script>parent.FM.view\((.*)\)</script>$', response, re.DOTALL)
	if m == None:
		print 'search: [Error] undefined pattern'
		return None

	json_str = m.group(1)
	
	data  = None
	try:
		data = json.loads(json_str)
	except:
		print 'search: [Error] json failed: %s'%(traceback.format_exc())
		return None

	soup = BeautifulSoup(data['html'], 'html.parser')

	wb_empty = soup.find('div', attrs={'class':'WB_empty'})
	if not wb_empty == None:
		print 'search: [Warning] result not found'
		return None

	wb_text = soup.find('div', attrs={'class':'WB_text'})
	if not wb_text:
		print 'search: [Error] WB_text not found'
		return None

	if not wb_text.text.startswith(key_word):
		print 'search: [Warning] first result mismatches'
		return None

	return wb_text.text

