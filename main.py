'''
This is the main file that runs the image display website.
'''
from flask import Flask, render_template
from gevent.pywsgi import WSGIServer
from datetime import datetime, timedelta
from threading import Timer
import random
import os
import requests
app = Flask(__name__, template_folder='.')
@app.route('/')
def index():
    images = get_images()
    return render_template('index.html', images=images)
def get_images():
    images = []
    current_time = datetime.now()
    image_dir = os.path.join(os.getcwd(), 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    # Check if images are already downloaded
    if not os.listdir(image_dir):
        download_images(image_dir)
    # Get images within the past week
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path):
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if current_time - modified_time < timedelta(days=7):
                images.append(filename)
            else:
                os.remove(file_path)  # Remove old images
    return images
def download_images(image_dir):
    url = 'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87'
    response = requests.get(url)
    if response.status_code == 200:
        try:
            image_urls = response.json()
            random.shuffle(image_urls)
            for i in range(10):
                image_url = image_urls[i]
                image_data = requests.get(image_url).content
                image_path = os.path.join(image_dir, f'image{i+1}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(image_data)
        except ValueError:
            print('Invalid JSON data')
    else:
        print('Failed to fetch image URLs')
def update_images():
    image_dir = os.path.join(os.getcwd(), 'images')
    download_images(image_dir)
    Timer(28800, update_images).start()
if __name__ == '__main__':
    update_images()
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
