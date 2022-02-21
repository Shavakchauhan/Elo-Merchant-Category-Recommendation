from flask import Flask, jsonify,request
import numpy as np
import pandas as pd


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)




@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    predicted_dataframe = pd.read_csv('predicted_targets.csv')
    predicted_dataframe.set_index('card_id',inplace=True)
    to_predict_list = request.form.to_dict()
    predicted_value = predicted_dataframe.loc[str(to_predict_list['card_id'])]['predicted_target']
    return {to_predict_list['card_id']:predicted_value}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
