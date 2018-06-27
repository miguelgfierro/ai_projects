import cherrypy
from paste.translogger import TransLogger
from flask import (Flask, request, abort, jsonify, make_response,
                   render_template)
import json
import os
import pandas as pd
import lightgbm as lgb
from utils import (BASELINE_MODEL, BAD_REQUEST, STATUS_OK, NOT_FOUND,
                   SERVER_ERROR, PORT, FRAUD_THRESHOLD, DATABASE_FILE,
                   TABLE_LOCATIONS)
from utils import connect_to_database, select_random_row


# app
app = Flask(__name__)
# define static folder for css, img, js
app.static_folder = 'static'


@app.errorhandler(BAD_REQUEST)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), BAD_REQUEST)


@app.errorhandler(NOT_FOUND)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), NOT_FOUND)


@app.errorhandler(SERVER_ERROR)
def server_error(error):
    return make_response(jsonify({'error': 'Server error'}), SERVER_ERROR)


@app.route('/')
def hello():
    return render_template('hello.html')


@app.route('/map')
def map():
    return render_template('index.html')


def manage_query(request):
    if not request.is_json:
        abort(BAD_REQUEST)
    dict_query = request.get_json()
    X = pd.DataFrame(dict_query, index=[0])
    return X


@app.route('/predict', methods=['POST'])
def predict():
    X = manage_query(request)
    y_pred = model.predict(X)[0]
    return make_response(jsonify({'fraud': y_pred}), STATUS_OK)


@app.route('/predict_map', methods=['GET', 'POST'])
def predict_map():
    X = manage_query(request)
    y_pred = model.predict(X)[0]

    row = select_random_row(conn, TABLE_LOCATIONS)
    location = {"title": row[0], "latitude": row[1], "longitude": row[2]}

    # FIXME: send real time input to the map
    locations_fair = [{
        "latitude": 28.6353,
        "longitude": 77.2250,
        "title": "New Delhi"
    }, {
        "latitude": -34.6118,
        "longitude": -58.4173,
        "title": "Buenos Aires"
    }]

    locations_fraud = [{
        "latitude": 34.05,
        "longitude": -118.24,
        "title": "Los Angeles"
    }, {
        "latitude": 35.6785,
        "longitude": 139.6823,
        "title": "Tokyo"
    }]

    return render_template('index.html',
                           locations_fair=locations_fair,
                           locations_fraud=locations_fraud)


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
        'server.thread_pool': 50,  # 10 is default
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


# Load the model as a global variable
model = lgb.Booster(model_file=BASELINE_MODEL)

# Connect to database
conn = connect_to_database(DATABASE_FILE)

if __name__ == "__main__":
    try:
        run_server()
    except:
        raise
    finally:
        conn.close()
