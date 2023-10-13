'''
This file contains the ImageManager class responsible for managing the images displayed on the website.
'''
import os
import glob
import random
import shutil
from datetime import datetime, timedelta
class ImageManager:
    def __init__(self):
        self.image_dir = 'images'
        self.max_images = 10
    def get_images(self):
        self.cleanup_images()
        images = self.get_random_images()
        return images
    def cleanup_images(self):
        current_time = datetime.now()
        for image_file in glob.glob(os.path.join(self.image_dir, '*.jpg')):
            file_time = datetime.fromtimestamp(os.path.getmtime(image_file))
            if current_time - file_time > timedelta(days=7):
                os.remove(image_file)
    def get_random_images(self):
        image_files = glob.glob(os.path.join(self.image_dir, '*.jpg'))
        random.shuffle(image_files)
        return image_files[:self.max_images]
