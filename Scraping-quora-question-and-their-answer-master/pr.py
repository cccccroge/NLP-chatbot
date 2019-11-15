from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import os.path
import re
import urllib.request
import json
import timeit
from multiprocessing import Pool
import multiprocessing

start = timeit.default_timer()

def add1(c):
    return c + 1

def get_soup(url,header):
    driver = webdriver.Chrome()
    r = driver.get(url)
    return BeautifulSoup(driver.page_source, 'html.parser')

header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
}

def parse(url):
    start = timeit.default_timer()
    soup = get_soup(url,header)
    html=urllib.request.urlopen(urllib.request.Request(url,headers=header)).read()
    urls=[]
    print(url)
    cnt=soup.find('div',{'class':'answer_count'})
    cnt_a=cnt.text
    print(cnt_a)
    hmsnmsl = soup.findAll("div", {"class": "ui_qtext_expanded"}, limit=None)
    for foo in hmsnmsl:
        #print(foo.span.text+'\n\n\n')
        query=foo.span.text
        urls.append(query)
    i = 0
    for tex in urls:
        while(os.path.exists("final" + str(i) + ".txt")):
            i += 1
        final = "final" + str(i) + ".txt"
        file=open(final,"w", encoding="utf-8")
        for s in tex:
            file.write(str(s))
        file.close()
    stop = timeit.default_timer()
    print(len(urls))
    print('\n'+'Time :')
    print(stop-start)
    print('\n\n')

with open("link.txt","r") as f:
    data_links=f.readlines()

#for i in range (0,len(data_links)):
#parse(data_links[0])

if __name__ == '__main__':
    processes = []
    for i in data_links:
        p = multiprocessing.Process(target=parse, args=(i,))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
        
stop = timeit.default_timer()
print('Time: ', stop - start)