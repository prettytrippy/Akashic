import requests
from bs4 import BeautifulSoup
import time
from text_utilities import safe_string
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

options = Options()
options.headless = True

banned_sites = ['google', 'youtube']

def clean_links(links):
    links = [link for link in links if link != None]
    for i in banned_sites:
        links = [l for l in links if i not in l]
    return links

def webscrape(query):
    driver = webdriver.Firefox(executable_path='../../../TrippUtilities/geckodriver', options=options)

    queries = query.split()
    query = "+".join(queries)

    # Make a request to Google search with the given query
    driver.get(f'https://www.google.com/search?q={query}')

    # Find all the links on the page
    links = driver.find_elements_by_css_selector('a')
    
    # Extract href attribute from each link
    links = [link.get_attribute('href') for link in links]
    links = clean_links(links)

    driver.quit()

    webtexts = []
    for link in links:
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.text, 'html.parser')
            webtexts.append(safe_string(soup.get_text(" ")))
        except:
            pass

    return webtexts
