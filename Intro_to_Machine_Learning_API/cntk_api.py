from flask import Flask
from flask import request, abort, jsonify, make_response, json, render_template
from cntk import load_model, combine
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np

STATUS_OK = 200
BAD_REQUEST = 400
app = Flask(__name__)


def read_image_from_request_base64(image_base64):
    img = Image.open(BytesIO(base64.b64decode(image_base64)))
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
    z_out = combine([model.outputs[3].owner])
    result = np.squeeze(z_out.eval({z_out.arguments[0]:[img]}))
    # Sort probabilities 
    prob_idx = np.argsort(result)[::-1][:number_results]
    pred = [labels[i] for i in prob_idx]
    return pred


@app.errorhandler(BAD_REQUEST)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), BAD_REQUEST)


@app.route('/api/v1/classify_image', methods=['POST'])
def classify_image():
    #if not request.json or 'image' not in request.json:
    #    abort(BAD_REQUEST)
    print("Request received")
    image_request = request.files['image']
    #img = read_image_from_request_base64(image_request)
    img = read_image_from_ioreader(image_request)
    resp = predict(model, img, labels, 5)
    return make_response(jsonify({'message': resp}), STATUS_OK)


@app.route('/')
def hello():
    return render_template('hello.html')


if __name__ == "__main__":
    model = load_model('ResNet_152.model')
    labels = read_synsets()
    app.run(debug=True, host='0.0.0.0')
