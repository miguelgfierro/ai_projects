from flask import (Flask, request, abort, jsonify, make_response,
                   render_template)
from flask_socketio import SocketIO, emit
import pandas as pd
import lightgbm as lgb
from utils import (BASELINE_MODEL, BAD_REQUEST, STATUS_OK, NOT_FOUND,
                   SERVER_ERROR, PORT, FRAUD_THRESHOLD, DATABASE_FILE,
                   TABLE_LOCATIONS)
from utils import connect_to_database, select_random_row


app = Flask(__name__)  # app
app.static_folder = 'static'  # define static folder for css, img, js
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


@socketio.on('connect', namespace='/fraud')
def test_connect():
    print('Client connected')


@socketio.on('disconnect', namespace='/fraud')
def test_disconnect():
    print('Client disconnected')


@socketio.on('my_ping', namespace='/fraud')
def ping_pong():
    """Receives my_pong from the client and sends my_pong from the server"""
    emit('my_pong')


@app.route('/health')
def health_check():
    """Health end point
    $ curl http://localhost:5000/health
    """
    socketio.emit('health_signal',
                  {'data': 'HEALTH CHECK', 'note': 'OK'},
                  broadcast=True,
                  namespace='/fraud')
    return make_response(jsonify({'health': 'OK'}), STATUS_OK)


def _manage_query(request):
    if not request.is_json:
        abort(BAD_REQUEST)
    dict_query = request.get_json()
    X = pd.DataFrame(dict_query, index=[0])
    return X


@app.route('/predict', methods=['POST'])
def predict():
    """Main API end point. Predicts whether a transaction
    is fraudulent or not using the LightGBM model.
    The easiest way to call this end point is to use the
    notebook fraud_detection.ipynb. The pyload is a json
    with 30 columns, corresponding to Time, V1-V28 and Amount.
    An example of a query is:
    $ curl -X POST -H "Content-type: application/json" http://127.0.0.1:5000/predict -d '{"Time": 57007.0, "V1": -1.2712441917143702, "V2": 2.46267526851135, "V3": -2.85139500331783, "V4": 2.3244800653477995, "V5": -1.37224488981369, "V6": -0.948195686538643, "V7": -3.06523436172054, "V8": 1.1669269478721105, "V9": -2.2687705884481297, "V10": -4.88114292689057, "V11": 2.2551474887046297, "V12": -4.68638689759229, "V13": 0.652374668512965, "V14": -6.17428834800643, "V15": 0.594379608016446, "V16": -4.8496923870965185, "V17": -6.53652073527011, "V18": -3.11909388163881, "V19": 1.71549441975915, "V20": 0.560478075726644, "V21": 0.652941051330455, "V22": 0.0819309763507574, "V23": -0.22134783119833895, "V24": -0.5235821592333061, "V25": 0.224228161862968, "V26": 0.756334522703558, "V27": 0.632800477330469, "V28": 0.25018709275719697, "Amount": 0.01}'
    Returns:
        resp (json): predicted fraud probability
    """
    X = _manage_query(request)
    y_pred = model.predict(X)[0]
    return make_response(jsonify({'fraud': y_pred}), STATUS_OK)


@app.route('/predict_map', methods=['POST'])
def predict_map():
    """Similar end point as /predict. The difference is that
    when a query is fraudulent, a websocket shows a location
    in the map in real-time. To visualize the map just go to
    http://localhost:5000/map.
    A query can be sent using the notebook fraud_detection.ipynb
    or curl with /predict_map end point and the same payload as
    before. An example of the query is:
    $ curl -X POST -H "Content-type: application/json" http://127.0.0.1:5000/predict_map -d '{"Time": 57007.0, "V1": -1.2712441917143702, "V2": 2.46267526851135, "V3": -2.85139500331783, "V4": 2.3244800653477995, "V5": -1.37224488981369, "V6": -0.948195686538643, "V7": -3.06523436172054, "V8": 1.1669269478721105, "V9": -2.2687705884481297, "V10": -4.88114292689057, "V11": 2.2551474887046297, "V12": -4.68638689759229, "V13": 0.652374668512965, "V14": -6.17428834800643, "V15": 0.594379608016446, "V16": -4.8496923870965185, "V17": -6.53652073527011, "V18": -3.11909388163881, "V19": 1.71549441975915, "V20": 0.560478075726644, "V21": 0.652941051330455, "V22": 0.0819309763507574, "V23": -0.22134783119833895, "V24": -0.5235821592333061, "V25": 0.224228161862968, "V26": 0.756334522703558, "V27": 0.632800477330469, "V28": 0.25018709275719697, "Amount": 0.01}'
    Returns:
        resp (json): predicted fraud probability
    """
    X = _manage_query(request)
    y_pred = model.predict(X)[0]
    print("Value predicted: {}".format(y_pred))
    if y_pred >= FRAUD_THRESHOLD:
        row = select_random_row(conn, TABLE_LOCATIONS)
        location = {"title": row[0], "latitude": row[1], "longitude": row[2]}
        print("New location: {}".format(location))
        socketio.emit('map_update', location, broadcast=True, namespace='/fraud')
    return make_response(jsonify({'fraud': y_pred}), STATUS_OK)


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
