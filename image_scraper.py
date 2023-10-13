'''
This file contains the ImageScraper class responsible for scraping images from the internet.
'''
import requests
import os
import random
import time
from bs4 import BeautifulSoup
class ImageScraper:
    def __init__(self):
        self.image_dir = 'images'
    def scrape_images(self):
        query = '熊'
        url = f'https://images.google.com/search?q={query}&tbm=isch'
        response = requests.get(url)
        response.raise_for_status()
        image_urls = self.extract_image_urls(response.text)
        self.download_images(image_urls)
    def extract_image_urls(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        image_urls = []
        for img in soup.find_all('img'):
            image_url = img.get('src')
            if image_url:
                image_urls.append(image_url)
        return image_urls
    def download_images(self, image_urls):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        for i, image_url in enumerate(image_urls):
            response = requests.get(image_url)
            if response.status_code == 200:
                image_path = os.path.join(self.image_dir, f'image_{i}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f'Failed to download image: {image_url}')
                continue  # Skip to the next image URL
            time.sleep(1)  # Delay to avoid overloading the server
