#!/usr/bin/env python  
# encoding: utf-8  
import urllib2  
import re  
import os  
import sys  
reload(sys)  
sys.setdefaultencoding("utf-8")  
  
def img_spider(name_file):  
    user_agent = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"  
    headers = {'User-Agent':user_agent}  
    #读取名单txt，生成包括所有物品的名单列表
    with open(name_file) as f:  
        name_list = [name.rstrip().decode('utf-8') for name in f.readlines()]  
        f.close()  
    #遍历每一个物品，保存在以该物品名字命名的文件夹中
    for name in name_list:  
        #生成文件夹（如果不存在的话）  
        if not os.path.exists('data/my_data/' + name):
            os.makedirs('data/my_data/' + name)
        for i in range(2):
            #修改range内数值n,可改变爬取数量为n*60
            try:
                num = (i+1)*60
                url = "http://image.baidu.com/search/avatarjson?tn=resultjsonavatarnew&ie=utf-8&word=" + name.replace(' ','%20') + "&cg=girl&rn=60&pn="+ str(num)
                req = urllib2.Request(url, headers=headers)
                res = urllib2.urlopen(req)
                page = res.read()
                #print page
                #因为JSON的原因，在浏览器页面按F12看到的，和你打印出来的页面内容是不一样的，所以匹配的是objURL
                img_srcs = re.findall('"objURL":"(.*?)"', page, re.S)
                print name,len(img_srcs)
            except:
                #如果访问失败，就跳到下一个继续执行代码，而不终止程序
                print name," error:"
                continue
            j = 1
            src_txt = ''

            #访问上述得到的图片路径，保存到本地
            for src in img_srcs:
                with open('data/my_data/' + name + '/'+name +'_' + str(num+j-60)+'.jpg','wb') as p:
                    try:
                        print "downloading No.%d"%(num+j-60)
                        req = urllib2.Request(src, headers=headers)
                        #设置一个urlopen的超时，如果3秒访问不到，就跳到下一个地址，防止程序卡在一个地方。
                        img = urllib2.urlopen(src,timeout=3)
                        p.write(img.read())
                    except:
                        print "No.%d error:"%(num+j-60)
                        p.close()
                        continue
                    p.close()
                src_txt = src_txt + src + '\n'
                if j==60:
                    break
                j = j+1

#主程序，读txt文件开始爬  
if __name__ == '__main__':  
    #name_file = "data/flower.txt"
    img_spider(name_file)  
