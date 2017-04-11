from flask import Flask
from flask import request, abort, jsonify, make_response, json, render_template
from cntk import load_model, combine
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import requests


STATUS_OK = 200
NOT_FOUND = 404
BAD_REQUEST = 400
BAD_PARAM = 450
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


def predict(model, image, labels, number_results = 5):
    #Crop and center the image
    img = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    #Transform the image for CNTK format
    img = np.array(img, dtype=np.float32)
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
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


#TODO: make custom error handler
#@app.errorhandler(BAD_PARAM)
#def bad_param(error=None):
#    return make_response(jsonify({'error': 'Bad parameter'}), BAD_PARAM)


@app.route('/api/v1/classify_image', methods=['POST'])
def classify_image():
    if 'image' in request.files:
        print("Image request")
        image_request = request.files['image']
        img = read_image_from_ioreader(image_request)
    elif 'url' in request.json: 
        print("JSON request: ", request.json)
        image_url = request.json['url']
        img = read_image_from_url(image_url)
    else:
        abort(BAD_REQUEST)
    resp = predict(model, img, labels, 5)
    return make_response(jsonify({'message': resp}), STATUS_OK)


@app.route('/')
def hello():
    return render_template('hello.html')


labels = read_synsets()
model = load_model('ResNet_152.model')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
