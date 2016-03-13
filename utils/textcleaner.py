from langconv import Converter

sc_converter = Converter('zh-hans')

def simplify_chinese(text):
	return sc_converter.convert(text)
