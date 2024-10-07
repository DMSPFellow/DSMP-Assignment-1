# -*- coding: utf-8 -*-
"""IDS-Assign1-1-Sobia.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11V-N1nGr6gpZiyg01JqXstSks1lV3Ab_
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

url = "https://webscraper.io/test-sites/e-commerce/allinone/computers/tablets"
r = requests.get(url)
r.content

soup = BeautifulSoup(r.text, "lxml")
soup

soup.prettify()

soup

names = soup.find_all("a", class_="title")
print(names)

Names = []
for i in names:
    print(i.text)
    Names.append(i.text)

prices = soup.find_all("h4", class_="pull-right price")
print(prices)

Prices = []
for i in prices:
    print(i.text)
    Prices.append(i.text)

description = soup.find_all("p", class_="description")
print(description)

Description = []
for i in description:
    print(i.text)
    Description.append(i.text)

reviews = soup.find_all("p", class_="pull-right")
print(reviews)

Reviews = []
for i in reviews:
    print(i.text)
    Reviews.append(i.text)

print(Names)
print(len(Names))

print(Prices)
print(len(Prices))

print(Description)
print(len(Description))

print(Reviews)
print(len(Reviews))

dataset = {
    "Name": Names,
    "Price": Prices,
    "Description": Description,
    "Reviews": Reviews
}

dataset

csv_file_path = "/Users/macbookpro/Desktop/ecommerce_data.csv"

df = pd.DataFrame(dataset)
df

df.to_csv(csv_file_path, index = False)