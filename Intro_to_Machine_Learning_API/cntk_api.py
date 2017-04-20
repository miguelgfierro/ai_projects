from flask import Flask
from flask import request, abort, jsonify, make_response, json, render_template
from cntk import load_model, combine
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import requests
import cherrypy
from paste.translogger import TransLogger
from config import DEVELOPMENT, PORT


STATUS_OK = 200
NOT_FOUND = 404
BAD_REQUEST = 400
SERVER_ERROR = 500
app = Flask(__name__)


def read_image_from_url(url):
    img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    return img


def read_image_from_ioreader(image_request):
    img = Image.open(BytesIO(image_request.read())).convert('RGB')
    return img


def read_synsets(filename='synsets.txt'):
    with open(filename, 'r') as f:
        synsets = [l.rstrip() for l in f]
        labels = [" ".join(l.split(" ")[1:]) for l in synsets]
    return labels


def get_preprocessed_image(my_image, mean_image):
    #Crop and center the image
    my_image = ImageOps.fit(my_image, (224, 224), Image.ANTIALIAS)
    #Transform the image for CNTK format
    my_image = np.array(my_image, dtype=np.float32)
    # RGB -> BGR
    bgr_image = my_image[:, :, ::-1] 
    image_data = np.ascontiguousarray(np.transpose(bgr_image, (2, 0, 1)))
    image_data -= mean_image
    return image_data


def predict(model, image, labels, number_results=5):
    img = get_preprocessed_image(image, 0)
    # Use last layer to make prediction
    arguments = {model.arguments[0]: [img]}
    result = model.eval(arguments)
    result = np.squeeze(result)
    # Sort probabilities 
    prob_idx = np.argsort(result)[::-1][:number_results]
    pred = [labels[i] for i in prob_idx]
    return pred


@app.errorhandler(BAD_REQUEST)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), BAD_REQUEST)


@app.errorhandler(NOT_FOUND)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), NOT_FOUND)


@app.errorhandler(SERVER_ERROR)
def server_error(error):
    return make_response(jsonify({'error': 'Server error'}), SERVER_ERROR)


@app.route('/api/v1/classify_image', methods=['POST'])
def classify_image():
    if 'image' in request.files:
        cherrypy.log("CHERRYPY LOG: Image request")
        image_request = request.files['image']
        img = read_image_from_ioreader(image_request)
    elif 'url' in request.json: 
        cherrypy.log("CHERRYPY LOG: JSON request: {}".format(request.json['url']))
        image_url = request.json['url']
        img = read_image_from_url(image_url)
    else:
        cherrypy.log("CHERRYPY LOG: Bad request")
        abort(BAD_REQUEST)
    resp = predict(model, img, labels, 5)
    return make_response(jsonify({'message': resp}), STATUS_OK)


@app.route('/')
def hello():
    return render_template('hello.html')


def run_server():
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload_on': True,
        'log.screen': True,
        'log.error_file': "cherrypy.log",
        'server.socket_port': PORT,
        'server.socket_host': '0.0.0.0',
        'server.thread_pool': 50, # 10 is default
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()

    
# Load synsets and model as global variables
labels = read_synsets()
model = load_model('ResNet_152.model')


if __name__ == "__main__":
    if DEVELOPMENT:
        app.run(debug=True)
    else:
        run_server()
