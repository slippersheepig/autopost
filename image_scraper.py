'''
This file contains the ImageScraper class responsible for scraping and managing images.
'''
import requests
import random
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
class ImageScraper:
    def __init__(self):
        self.images = []
        self.last_update = None
    def get_images(self):
        if self.last_update is None or datetime.now() - self.last_update > timedelta(hours=8):
            self.update_images()
        return self.images
    def update_images(self):
        self.images = self.scrape_images()
        self.last_update = datetime.now()
    def scrape_images(self):
        url = 'https://images.google.com'
        query = '熊'
        search_url = f'{url}/search?q={query}'
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_elements = soup.find_all('img')
        image_urls = []
        for img in image_elements:
            if '熊' in img.get('alt', '').lower() or '熊' in img.get('title', '').lower() or '熊' in img.get_text().lower():
                image_urls.append(img['src'])
        random.shuffle(image_urls)
        return image_urls[:10]
