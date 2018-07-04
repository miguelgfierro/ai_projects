from flask import (Flask, request, abort, jsonify, make_response,
                   render_template)
from flask_socketio import SocketIO, emit
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
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


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


@socketio.on('connect', namespace='/fraud')
def test_connect():
    print('Client connected')


@socketio.on('disconnect', namespace='/fraud')
def test_disconnect():
    print('Client disconnected')


@app.route('/predict_map', methods=['POST'])
def predict_map():
    X = manage_query(request)
    y_pred = model.predict(X)[0]
    print("Value predicted: {}".format(y_pred))
    row = select_random_row(conn, TABLE_LOCATIONS)
    location = {"title": row[0], "latitude": row[1], "longitude": row[2]}
    print("New location: {}".format(location))
    socketio.emit('map_update', location, broadcast=True)
    # fake_loc = {'title': 'Tebessa', 
    #             'latitude': 35.41043418, 
    #             'longitude': 8.120010537}
    # socketio.emit('map_update', fake_loc, broadcast=True)            
    return make_response(jsonify({'fraud': y_pred}), STATUS_OK)



# Receives my_pong from the client and sends my_pong from the server
@socketio.on('my_ping', namespace='/fraud')
def ping_pong():
    emit('my_pong')
#     row = select_random_row(conn, TABLE_LOCATIONS)
#     location = {"title": row[0], "latitude": row[1], "longitude": row[2]}
#     print("New location: {}".format(location))
#     emit('map_update', location, broadcast=True)


# Load the model as a global variable
model = lgb.Booster(model_file=BASELINE_MODEL)

# Connect to database
conn = connect_to_database(DATABASE_FILE)

if __name__ == "__main__":
    try:
        print("Server started")
        socketio.run(app, debug=True)
    except:
        raise
    finally:
        print("Stop procedure")
        conn.close()
