from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import json
from datetime import datetime
from collections import OrderedDict
from bs4 import BeautifulSoup

url='https://www.quora.com/search?q=typical+day'
#browser = webdriver.Firefox()
chromedriver = r'''C:\Users\Divakar Singh\Downloads\Compressed\chromedriver'''
browser=webdriver.PhantomJS()
#browser=webdriver.Chrome(chromedriver)
chrome_options = Options()
browser.get(url)
SCROLL_PAUSE_TIME = 0.5
# Get scroll height
last_height = browser.execute_script("return document.body.scrollHeight")

for i in range(0,3):
    # Scroll down to bottom
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Wait to load page
    SCROLL_PAUSE_TIME=SCROLL_PAUSE_TIME*2
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = browser.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        # try again (can be removed)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Calculate new scroll height and compare with last scroll height
            new_height = browser.execute_script("return document.body.scrollHeight")
            # check if the page height has remained the same
            if new_height == last_height:
                # if so, you are done
                break
    last_height = new_height
    print (i)
print(9)
urls=[]
file=open('link.txt','w')
html = browser.page_source
soup=BeautifulSoup(html,'html.parser')
for foo in soup.findAll("div", {"class": "pagedlist_item"}):
    print(foo.a.text)
    query=foo.a.text
    query= query.split()
    query='-'.join(query)
    urls.append("https://www.quora.com/"+query)
print(8)
for s in urls:
    file.write(str(s)+'\n')
file.close()
file=open("img.html","w")
file.write(str(html.encode('utf-8')))
file.close()
print(urls)
print(len(urls))
