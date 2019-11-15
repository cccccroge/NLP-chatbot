# Scraping-quora-question-and-their-answer
Here we are scrapping question related to some tag .Then according to the question we scrapped we will find answer  every answer related to that question. 

In question.py we stored question related to tag="gre" and then store the questions link in link.txt .

Then by using link.txt we will get every possible result of our question link and store their result in final1.txt here we will use multiprocessing in our program the technique used is Pool .So we will fwtch answer for almost 60 question and their about 5 answer and the whole process will take about 100 seconds.

Pre-requisite used :-

->selenium

->bs4

->BeautifulSoup

->url

->urllib

->PhantomJS

->chromedriver

->webdriver

->multiprocessing

->Pool

For running the code (in command prompt):-

->python question.py

->python pr.py
