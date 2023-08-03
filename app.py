from flask import Flask, request, send_file, jsonify
from flask_cors import CORS, cross_origin
#import tensorflow as tf
#from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

import json

import env

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.secret_key = "chronicare123"
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


def composeReply(status, message, payload = None):
    reply = {}
    reply["SENDER"] = "CHRONICARE AI"
    reply["STATUS"] = status
    reply["MESSAGE"] = message
    reply["PAYLOAD"] = payload
    return jsonify(reply)


@app.route("/diabetes", methods=['POST'])
def diabetes():
    age = request.form.get("age")
    gender = request.form.get("gender")
    polyuria = request.form.get("polyuria")
    polydipsia = request.form.get("polydipsia")
    sudden_weight_loss = request.form.get("sudden_weight_loss")
    polyphagia = request.form.get("polyphagia")
    delayed_healing = request.form.get("delayed_healing")
    obesity = request.form.get("obesity")
    
    #model = load_model('diabetes/diabetes.h5')
    model = load_model(env.fullPath + '\\diabetes\\diabetes.h5')
    #model = load_model(env.fullPath + '/ml/ml.h5')
    
    new_data_dict = {
        'age':                  [int(age)],
        'gender':               [int(gender)],
        'polyuria':             [int(polyuria)],
        'polydipsia':           [int(polydipsia)],
        'sudden_weight_loss':   [int(sudden_weight_loss)],
        'polyphagia':           [int(polyphagia)],
        'delayed_healing':      [int(delayed_healing)],
        'obesity':              [int(obesity)]
    }

    data = json.load(open('diabetes/diabetes.json'))

    new_data = pd.DataFrame(new_data_dict)
    new_data['age'] = (new_data['age'] - data["age_min"]) / (data["age_max"] - data["age_min"])

    predictions = model.predict(new_data)
    print("============================================================")
    finalPredict = predictions[0].tolist()[0]
    print(finalPredict)
    
    probabilities_positive_class = predictions[:, 0]
    probabilities_negative_class = 1 - probabilities_positive_class
    confident_percent_positive_class = probabilities_positive_class * 100
    confident_percent_negative_class = probabilities_negative_class * 100

    confident_p = round(confident_percent_positive_class.tolist()[0], 2)
    confident_n = round(confident_percent_negative_class.tolist()[0], 2)

    returnData = {
        "PREDICTION" : finalPredict,
        "CONFIDENT" : confident_p if finalPredict > 0.5 else confident_n,
        "CONFIDENT_POSITIVE" : confident_p,
        "CONFIDENT_NEGATIVE" : confident_n,
    }

    return composeReply("SUCCESS", "Prediksi", returnData)


@app.route("/diabetes2", methods=['POST'])
def diabetes2():
    age = request.form.get("age")
    gender = request.form.get("gender")
    polyuria = request.form.get("polyuria")
    polydipsia = request.form.get("polydipsia")
    sudden_weight_loss = request.form.get("sudden_weight_loss")
    polyphagia = request.form.get("polyphagia")
    delayed_healing = request.form.get("delayed_healing")
    obesity = request.form.get("obesity")
    
    new_data_dict = {
        'age':                  [int(age)],
        'gender':               [int(gender)],
        'polyuria':             [int(polyuria)],
        'polydipsia':           [int(polydipsia)],
        'sudden_weight_loss':   [int(sudden_weight_loss)],
        'polyphagia':           [int(polyphagia)],
        'delayed_healing':      [int(delayed_healing)],
        'obesity':              [int(obesity)]
    }

    # arr = [
    #    58,	1,	0,	0,	0,	1,	1,	1
    # ]
    # new_data_dict = {'age':[arr[0]],'gender':[arr[1]],'polyuria':[arr[2]],'polydipsia':[arr[3]],'sudden_weight_loss':[arr[4]],'polyphagia':[arr[5]],'delayed_healing':[arr[6]],'obesity':[arr[7]]
    # }

    if env.development == "local":
        filePath = '\\diabetes\\'
    elif env.development == "doscom":
        filePath = "/diabetes/"

    loaded_model = pickle.load(open(env.fullPath +  filePath + "diabetes.sav", 'rb'))
    features = pd.DataFrame(new_data_dict, index=[0])

    #features = pd.get_dummies(features, columns=['gender'])

    with open(env.fullPath +  filePath + "scaler.pkl", 'rb') as file:
        scaler = pickle.load(file)
    features['age'] = scaler.transform(features[['age']])

    prediction = loaded_model.predict(features)
    finalPredict = prediction[0].tolist()
    probability = loaded_model.predict_proba(features)

    confidenceN = probability[0][0]
    confidenceP = probability[0][1]

    print("==================================================")
    print("confidence of 0: " + str(confidenceN))
    print("confidence of 1: " + str(confidenceP))
    print("prediction : " + str(finalPredict))
    print(features[['age']])

    returnData = {
        "PREDICTION" : finalPredict,
        "CONFIDENT" : confidenceP if finalPredict > 0.5 else confidenceN,
        "CONFIDENT_POSITIVE" : confidenceP,
        "CONFIDENT_NEGATIVE" : confidenceN,
    }

    return composeReply("SUCCESS", "Prediksi", returnData)


if __name__ == '__main__':
    app.run(host = env.runHost, port = env.runPort, debug = env.runDebug)