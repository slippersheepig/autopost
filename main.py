'''
This is the main file that runs the image display website.
'''
from flask import Flask, render_template
from gevent.pywsgi import WSGIServer
from image_manager import ImageManager
app = Flask(__name__, template_folder='.')
image_manager = ImageManager()
@app.route('/')
def index():
    images = image_manager.get_images()
    return render_template('index.html', images=images)
if __name__ == '__main__':
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
