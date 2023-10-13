'''
This file contains the ImageManager class responsible for managing the images.
'''
import os
import random
import requests
import time
class ImageManager:
    def __init__(self):
        self.images_dir = 'images'
        self.image_urls = [
            'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87/1-500',
            'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87/501-1000',
            'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87/1001-1500',
            'https://h.sheepgreen.top/%E5%9B%BE%E7%89%87/1501-2000'
        ]
        self.update_interval = 8 * 60 * 60  # 8 hours
        self.cleanup_interval = 7 * 24 * 60 * 60  # 1 week
    def get_images(self):
        self._update_images()
        self._cleanup_images()
        return self._get_image_files()
    def _update_images(self):
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        current_time = time.time()
        last_update_time = self._get_last_update_time()
        if current_time - last_update_time >= self.update_interval:
            self._download_images()
            self._set_last_update_time(current_time)
    def _download_images(self):
        for url in self.image_urls:
            response = requests.get(url)
            if response.status_code == 200:
                image_file = os.path.join(self.images_dir, f'{random.randint(1, 100000)}.jpg')
                with open(image_file, 'wb') as file:
                    file.write(response.content)
    def _cleanup_images(self):
        if not os.path.exists(self.images_dir):
            return
        current_time = time.time()
        for image_file in self._get_image_files():
            file_path = os.path.join(self.images_dir, image_file)
            if current_time - os.path.getmtime(file_path) >= self.cleanup_interval:
                os.remove(file_path)
    def _get_image_files(self):
        if not os.path.exists(self.images_dir):
            return []
        return sorted(os.listdir(self.images_dir))
    def _get_last_update_time(self):
        last_update_file = os.path.join(self.images_dir, 'last_update.txt')
        if os.path.exists(last_update_file):
            with open(last_update_file, 'r') as file:
                return float(file.read())
        return 0
    def _set_last_update_time(self, timestamp):
        last_update_file = os.path.join(self.images_dir, 'last_update.txt')
        with open(last_update_file, 'w') as file:
            file.write(str(timestamp))
