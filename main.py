'''
This is the main file that runs the image display website.
'''
from flask import Flask, render_template
from gevent.pywsgi import WSGIServer
from datetime import datetime, timedelta
import random
import os
import requests
from bs4 import BeautifulSoup
import time
app = Flask(__name__, template_folder='.')
@app.route('/')
def index():
    # Get the list of image URLs
    image_urls = get_image_urls()
    # Render the template with the image URLs
    return render_template('index.html', image_urls=image_urls)
def get_image_urls():
    # Check if the image directory exists, create it if not
    if not os.path.exists('images'):
        os.makedirs('images')
    # Get the current time
    current_time = datetime.now()
    # Check if the images need to be updated
    if not os.path.exists('last_updated.txt'):
        update_images()
        with open('last_updated.txt', 'w') as f:
            f.write(str(current_time))
    else:
        with open('last_updated.txt', 'r') as f:
            last_updated = datetime.strptime(f.read(), '%Y-%m-%d %H:%M:%S.%f')
        if current_time - last_updated >= timedelta(hours=8):
            update_images()
            with open('last_updated.txt', 'w') as f:
                f.write(str(current_time))
    # Get the list of image files
    image_files = os.listdir('images')
    # Get the image URLs
    image_urls = []
    for image_file in image_files:
        image_urls.append(f'/images/{image_file}')
    return image_urls
def update_images():
    # Clear the images directory
    for image_file in os.listdir('images'):
        os.remove(os.path.join('images', image_file))
    # Get the image URLs from the specified sources
    sources = [
        'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87/1-500',
        'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87/501-1000',
        'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87/1001-1500',
        'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87/1501-2000'
    ]
    for source in sources:
        response = requests.get(source)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tags = soup.find_all('img')
        for image_tag in image_tags:
            image_url = image_tag['src']
            image_response = requests.get(image_url)
            # Save the image file
            image_file = os.path.join('images', os.path.basename(image_url))
            with open(image_file, 'wb') as f:
                f.write(image_response.content)
if __name__ == '__main__':
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
