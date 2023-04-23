from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from ensemble_models import make_ensemble_preds

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({'message': "Hello, please use POST Method to get prediction."})


@app.route('/', methods=['POST'])
def predict():
    if 'input_data' not in request.json:
        return jsonify({'error': "Input data was not provided"})

    input_data = request.json['input_data']
    if not isinstance(input_data, list):
        return jsonify({'error': "Input data must be an array."})
    if len(input_data) < 7:
        return jsonify({'error': "Input array must be at least 7 days long"})

    input_data = input_data[-7:]
    input_data = np.array(request.json['input_data'])
    input_data = input_data.reshape((1, input_data.shape[0], 1))
    input_data = tf.constant(input_data, dtype=tf.float32)
    prediction = make_ensemble_preds(input_data)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')