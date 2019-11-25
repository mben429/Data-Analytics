from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style
import requests


def scrape_url(curr_url):
    uClient = uReq(curr_url)
    page_html = uClient.read()
    uClient.close()

    page_soup = soup(page_html, "html5lib")
    tr = list(page_soup.find_all("tr"))

    print(tr[14:16])

scrape_url("https://en.wikipedia.org/wiki/History_of_rugby_union_matches_between_England_and_New_Zealand")
