'''
This is the main file that runs the image display website.
'''
from flask import Flask, render_template
from gevent.pywsgi import WSGIServer
from image_scraper import ImageScraper
app = Flask(__name__, template_folder='.')
@app.route('/')
def display_images():
    image_scraper = ImageScraper()
    images = image_scraper.get_images()
    return render_template('index.html', images=images)
if __name__ == '__main__':
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
