#/usr/bin/env python
#coding=utf8
 
# import u
# py2
from urllib2 import urlopen, Request, quote
import md5

# py3
# import http.client
# from urllib.request import urlopen, Request, quote
# from hashlib import md5

import random
import json


appid = '20160715000025271'
secretKey = 'NZBmOnjrNLDxThhrJEUA'

def translation_en_2_zh(q):
    myurl = '/api/trans/vip/translate'
    # q = 'A group of people sitting about table eat something.'
    fromLang = 'en'
    toLang = 'zh'
    salt = random.randint(32768, 65536)

    sign = appid+q+str(salt)+secretKey
    m1 = md5.new()
    m1.update(sign)
    sign = m1.hexdigest()
    myurl = myurl+'?appid='+appid+'&q='+quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign

    try:
        req = Request('http://api.fanyi.baidu.com' + myurl)
        response = urlopen(req)
        the_page = response.read()
        json_data = json.loads(the_page)
        return (json_data["trans_result"][0]["dst"])

        #response是HTTPResponse对象
    except Exception as e:
        print(e)

def translation_en_2_zh_dummmy(q):
    return u'一群人站在领奖台上接受颁奖'


if __name__ == '__main__':
    print(translation_en_2_zh('a group of people standing in a field'))
