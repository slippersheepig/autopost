'''
This is the main Python file that runs the Flask application.
'''
from flask import Flask, render_template
from gevent.pywsgi import WSGIServer
from threading import Timer
import random
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html', random=random)
@app.route('/bear.txt')
def bear():
    random_color = '#%06x' % random.randint(0, 0xFFFFFF)
    return render_template('bear.txt', random_color=random_color)
if __name__ == '__main__':
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
