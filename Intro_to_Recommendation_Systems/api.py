import cherrypy
from paste.translogger import TransLogger
from flask import Flask, request, abort, jsonify, make_response
import json
import os
from utils import decode_string
from DeepRecommender.reco_encoder.data import input_layer_api, input_layer
from DeepRecommender.reco_encoder.model import model
import torch
from torch.autograd import Variable 
from parameters import *


# app
app = Flask(__name__)
BAD_REQUEST = 400
STATUS_OK = 200
NOT_FOUND = 404
SERVER_ERROR = 500


@app.errorhandler(BAD_REQUEST)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), BAD_REQUEST)


@app.errorhandler(NOT_FOUND)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), NOT_FOUND)


@app.errorhandler(SERVER_ERROR)
def server_error(error):
    return make_response(jsonify({'error': 'Server Internal Error'}), SERVER_ERROR)


def run_server():
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload_on': True,
        'log.screen': True,
        'log.access_file': 'access.log',
        'log.error_file': "cherrypy.log",
        'server.socket_port': 5000,
        'server.socket_host': '0.0.0.0',
        'server.thread_pool': 50, # 10 is default
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


def load_model_weights(model_architecture, weights_path):
    if os.path.isfile(weights_path):
        cherrypy.log("CHERRYPYLOG Loading model from: {}".format(weights_path))
        model_architecture.load_state_dict(torch.load(weights_path))
    else:
        raise ValueError("Path not found {}".format(weights_path))

        
def load_recommender(vector_dim, hidden, activation, dropout, weights_path):

    rencoder_api = model.AutoEncoder(layer_sizes=[vector_dim] + [int(l) for l in hidden.split(',')],
                               nl_type=activation,
                               is_constrained=False,
                               dp_drop_prob=dropout,
                               last_layer_activations=False)
    load_model_weights(rencoder_api, weights_path) 
    rencoder_api.eval()
    rencoder_api = rencoder_api.cuda()
    return rencoder_api

        
def load_train_data(data_dir):    
    params = dict()
    params['batch_size'] = 1
    params['data_dir'] =  data_dir
    params['major'] = 'users'
    params['itemIdInd'] = 1
    params['userIdInd'] = 0
    cherrypy.log("CHERRYPYLOG Loading training data")
    data_layer = input_layer.UserItemRecDataProvider(params=params)
    return data_layer


def manage_query(dict_query, data_layer):
    params = dict()
    params['batch_size'] = 1
    params['data_dict'] =  dict_query
    params['major'] = 'users'
    params['itemIdInd'] = 1
    params['userIdInd'] = 0
    data_api = input_layer_api.UserItemRecDataProviderAPI(params=params,
                                                        user_id_map=data_layer.userIdMap,
                                                        item_id_map=data_layer.itemIdMap)
    cherrypy.log("CHERRYPYLOG Input data: {}".format(data_api.data))
    data_api.src_data = data_layer.data
    return data_api


def evaluate_model(rencoder_api, data_api):   
    result = dict()
    for i, ((out, src), major_ind) in enumerate(data_api.iterate_one_epoch_eval(for_inf=True)):
        inputs = Variable(src.cuda().to_dense())
        targets_np = out.to_dense().numpy()[0, :]
        non_zeros = targets_np.nonzero()[0].tolist() 
        outputs = rencoder_api(inputs).cpu().data.numpy()[0, :]
        for ind in non_zeros:
            result[ind] = outputs[ind]
    return result

    
@app.route("/")
def index():
    return "Yeah, yeah, I highly recommend it"


@app.route("/recommend", methods=['POST'])
def recommend():
    if not request.is_json:
        abort(BAD_REQUEST)
    dict_query = request.get_json()
    dict_query = dict((decode_string(k), decode_string(v)) for k, v in dict_query.items())
    data_api = manage_query(dict_query, data_layer)
    result = evaluate_model(rencoder_api, data_api)
    cherrypy.log("CHERRYPYLOG Result: {}".format(result))
    result = dict((str(k), str(v)) for k,v in result.items())
    return make_response(jsonify(result), STATUS_OK)

  
#Load data and model as global variables
data_layer = load_train_data(TRAIN)
rencoder_api = load_recommender(data_layer.vector_dim, HIDDEN, ACTIVATION, DROPOUT, MODEL_PATH)


if __name__ == "__main__":
    run_server()
    
    