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

    return composeReply("SUCCESS", "Prediksi Diabetes", returnData)


@app.route("/hipertensi", methods=['POST'])
def hipertensi():
    age = request.form.get("age")
    bmi = request.form.get("bmi")
    sex = request.form.get("sex")
    physical_activity = request.form.get("physical_activity")
    salt_content_in_the_diet = request.form.get("salt_content_in_the_diet")
    level_of_stress = request.form.get("level_of_stress")
    adrenal_and_thyroid_disorders = request.form.get("adrenal_and_thyroid_disorders")
    
    new_data_dict = {
        'Age':                  [(age)],
        'BMI':                 [(bmi)],
        'Sex':             [(sex)],
        'Physical_activity':      [(physical_activity)],
        'salt_content_in_the_diet':           [(salt_content_in_the_diet)],
        'Level_of_Stress':   [(level_of_stress)],
        'Adrenal_and_thyroid_disorders':           [(adrenal_and_thyroid_disorders)],
    }

    if env.development == "local":
        filePath = '\\hipertensi\\'
    elif env.development == "doscom":
        filePath = "/hipertensi/"

    loaded_model = pickle.load(open(env.fullPath +  filePath + "hipertensi.sav", 'rb'))
    df = pd.DataFrame(new_data_dict, index=[0])

    with open(env.fullPath +  filePath + "scaler_bmi.pkl", 'rb') as file:
        scaler_bmi = pickle.load(file)
    df['BMI'] = scaler_bmi.transform(df[['BMI']])

    with open(env.fullPath +  filePath + "scaler_age.pkl", 'rb') as file:
        scaler_age = pickle.load(file)
    df['Age'] = scaler_age.transform(df[['Age']])

    prediction = loaded_model.predict(df)
    finalPredict = prediction[0].tolist()
    probability = loaded_model.predict_proba(df)

    confidenceN = probability[0][0]
    confidenceP = probability[0][1]

    print("==================================================")
    print("prediction : " + str(finalPredict))
    print("probability : " + str(probability))
    print("confidence of 0: " + str(confidenceN))
    print("confidence of 1: " + str(confidenceP))

    returnData = {
        "PREDICTION" : finalPredict,
        "CONFIDENT" : max([confidenceN, confidenceP]),
        "CONFIDENT_POSITIVE" : confidenceP,
        "CONFIDENT_NEGATIVE" : confidenceN,
    }

    return composeReply("SUCCESS", "Prediksi Hipertensi", returnData)


@app.route("/kanker", methods=['POST'])
def kanker():
    age = request.form.get("age")
    gender = request.form.get("gender")
    air_pollution = request.form.get("air_pollution")
    occupational_hazards = request.form.get("occupational_hazards")
    genetic_risk = request.form.get("genetic_risk")
    smoking = request.form.get("smoking")
    passive_smoker = request.form.get("passive_smoker")
    chest_pain = request.form.get("chest_pain")
    coughing_of_blood = request.form.get("coughing_of_blood")
    weight_loss = request.form.get("weight_loss")
    shortness_of_breath = request.form.get("shortness_of_breath")
    wheezing = request.form.get("wheezing")
    dry_cough = request.form.get("dry_cough")
    
    new_data_dict = {
        'Age': [int(age)],
        'Gender': [int(gender)],
        'Air Pollution': [int(air_pollution)],
        'OccuPational Hazards': [int(occupational_hazards)],
        'Genetic Risk': [int(genetic_risk)],
        'Smoking': [int(smoking)],
        'Passive Smoker': [int(passive_smoker)],
        'Chest Pain': [int(chest_pain)],
        'Coughing of Blood': [int(coughing_of_blood)],
        'Weight Loss': [int(weight_loss)],
        'Shortness of Breath': [int(shortness_of_breath)],
        'Wheezing': [int(wheezing)],
        'Dry Cough': [int(dry_cough)],
    }

    # arr = [
    #    35,	1,	4,	5,	5,	2,	3,	4,	8,	7,	9,	2,	7

    # ]
    # new_data_dict = {'Age':[arr[0]],'Gender':[arr[1]],'Air Pollution':[arr[2]],'OccuPational Hazards':[arr[3]],'Genetic Risk':[arr[4]],'Smoking':[arr[5]],'Passive Smoker':[arr[6]],'Chest Pain':[arr[7]],'Coughing of Blood':[arr[8]],'Weight Loss':[arr[9]],'Shortness of Breath':[arr[10]],'Wheezing':[arr[11]],'Dry Cough':[arr[12]]
    # }

    if env.development == "local":
        filePath = '\\kanker\\'
    elif env.development == "doscom":
        filePath = "/kanker/"

    loaded_model = pickle.load(open(env.fullPath +  filePath + "kanker.sav", 'rb'))
    features = pd.DataFrame(new_data_dict, index=[0])

    with open(env.fullPath +  filePath + "scaler_age.pkl", 'rb') as file:
        scaler = pickle.load(file)
    features['Age'] = scaler.transform(features[['Age']])

    prediction = loaded_model.predict(features)
    finalPredict = prediction[0]
    probability = loaded_model.predict_proba(features)

    prob0 = probability[0][0]
    prob1 = probability[0][1]
    prob2 = probability[0][2]

    print("==================================================")
    print("prediction : " + str(finalPredict))
    print("probability: " + str(probability))
    print("probability 0: " + str(prob0))
    print("probability 1: " + str(prob1))
    print("probability 2: " + str(prob2))

    returnData = {
        "PREDICTION" : finalPredict,
        "PROBABILITY" : max([prob0, prob1, prob2]),
        "PROBABILITY_LOW" : prob0,
        "PROBABILITY_MEDIUM" : prob1,
        "PROBABILITY_HIGH" : prob2,
    }

    return composeReply("SUCCESS", "Prediksi Kanker", returnData)


@app.route("/jantung", methods=['POST'])
def jantung():
    bmi = request.form.get("bmi")
    smoking = request.form.get("smoking")
    alcohol_drinking = request.form.get("alcohol_drinking")
    stroke = request.form.get("stroke")
    physical_health = request.form.get("physical_health")
    mental_health = request.form.get("mental_health")
    sex = request.form.get("sex")
    age_category = request.form.get("age_category")
    diabetic = request.form.get("diabetic")
    physical_activity = request.form.get("physical_activity")
    gen_health = request.form.get("gen_health")
    sleep_time = request.form.get("sleep_time")
    kidney_disease = request.form.get("kidney_disease")
    
    new_data_dict = {
        'BMI':                  [(bmi)],
        'Smoking':                 [(smoking)],
        'AlcoholDrinking':             [(alcohol_drinking)],
        'Stroke':           [(stroke)],
        'PhysicalHealth':   [(physical_health)],
        'MentalHealth':           [(mental_health)],
        'Sex':      [(sex)],
        'AgeCategory':         [(age_category)],
        'Diabetic':              [(diabetic)],
        'PhysicalActivity':      [(physical_activity)],
        'GenHealth':              [(gen_health)],
        'SleepTime':              [(sleep_time)],
        'KidneyDisease':        [(kidney_disease)],
    }

    if env.development == "local":
        filePath = '\\jantung\\'
    elif env.development == "doscom":
        filePath = "/jantung/"

    loaded_model = pickle.load(open(env.fullPath +  filePath + "jantung.sav", 'rb'))
    df = pd.DataFrame(new_data_dict, index=[0])

    df["BMI"] = df["BMI"].replace(',', '.', regex=True)
    df["Smoking"] = df['Smoking'].replace(['Yes', 'No'], ["1", "0"])
    df["AlcoholDrinking"] = df['AlcoholDrinking'].replace(['Yes', 'No'], ["1", "0"])
    df["Stroke"] = df['Stroke'].replace(['Yes', 'No'], ["1", "0"])
    df["Sex"] = df['Sex'].replace(['Male', 'Female'], ["1", "0"])

    df["PhysicalActivity"] = df['PhysicalActivity'].replace(['Yes', 'No'], ["1", "0"])
    df["KidneyDisease"] = df['KidneyDisease'].replace(['Yes', 'No'], ["1", "0"])

    import category_encoders as ce
    # Buat objek ordinal encoder
    ordinal_encoder = ce.OrdinalEncoder(cols=['AgeCategory'], mapping=[{'col': 'AgeCategory', 'mapping': {'18-24': 1, '25-29': 2, '30-34': 3, '35-39': 4, '40-44': 5, '45-49': 6, '50-54': 7, '55-59': 8, '60-64': 9, '65-69': 10, '70-74': 11, '75-79': 12, '80 or older': 13}}])
    # Lakukan ordinal encoding pada kolom "AgeCategory"
    df = ordinal_encoder.fit_transform(df)

    ordinal_encoder = ce.OrdinalEncoder(cols=['GenHealth'], mapping=[{'col': 'GenHealth', 'mapping': {'Poor': 1, 'Fair': 2, 'Good': 3, 'Very good': 4, 'Excellent': 5}}])
    # Lakukan ordinal encoding pada kolom "GenHealth"
    df = ordinal_encoder.fit_transform(df)

    ordinal_encoder = ce.OrdinalEncoder(cols=['Diabetic'], mapping=[{'col': 'Diabetic', 'mapping': {'No': 1, 'No, borderline diabetes': 2, 'Yes (during pregnancy)': 3, 'Yes': 4}}])
    # Lakukan ordinal encoding pada kolom "Diabetic"
    df = ordinal_encoder.fit_transform(df)

    with open(env.fullPath +  filePath + "scaler_bmi.pkl", 'rb') as file:
        scaler = pickle.load(file)
    df['BMI'] = scaler.transform(df[['BMI']])

    prediction = loaded_model.predict(df)
    finalPredict = prediction[0]
    probability = loaded_model.predict_proba(df)

    confidenceN = probability[0][0]
    confidenceP = probability[0][1]

    print("==================================================")
    print("prediction : " + str(finalPredict))
    print("probability : " + str(probability))
    print("confidence of 0: " + str(confidenceN))
    print("confidence of 1: " + str(confidenceP))

    returnData = {
        "PREDICTION" : finalPredict,
        "CONFIDENT" : max([confidenceN, confidenceP]),
        "CONFIDENT_POSITIVE" : confidenceP,
        "CONFIDENT_NEGATIVE" : confidenceN,
    }

    return composeReply("SUCCESS", "Prediksi Kanker Paru-Paru", returnData)


@app.route("/stroke", methods=['POST'])
def stroke():
    gender = request.form.get("gender")
    age = request.form.get("age")
    hypertension = request.form.get("hypertension")
    heart_disease = request.form.get("heart_disease")
    work_type = request.form.get("work_type")
    avg_glucose_level = request.form.get("avg_glucose_level")
    bmi = request.form.get("bmi")
    smoking_status = request.form.get("smoking_status")
    
    new_data_dict = {
        'gender':                  [(gender)],
        'age':                 [(age)],
        'hypertension':             [(hypertension)],
        'heart_disease':           [(heart_disease)],
        'work_type':   [(work_type)],
        'avg_glucose_level':           [(avg_glucose_level)],
        'bmi':      [(bmi)],
        'smoking_status':         [(smoking_status)]
    }

    if env.development == "local":
        filePath = '\\stroke\\'
    elif env.development == "doscom":
        filePath = "/stroke/"

    loaded_model = pickle.load(open(env.fullPath +  filePath + "stroke.sav", 'rb'))
    df = pd.DataFrame(new_data_dict, index=[0])

    df["bmi"] = df["bmi"].replace(',', '.', regex=True)
    df["hypertension"] = df['hypertension'].replace(['Yes', 'No'], ["1", "0"])
    df["heart_disease"] = df['heart_disease'].replace(['Yes', 'No'], ["1", "0"])
    df["gender"] = df['gender'].replace(['Male', 'Female'], ["0", "1"])
    
    with open(env.fullPath +  filePath + 'le_work.pkl', 'rb') as f:
        le_work = pickle.load(f)
    df['work_type'] = le_work.transform(df['work_type'])

    with open(env.fullPath +  filePath + 'le_smoking.pkl', 'rb') as f:
        le_smoking = pickle.load(f)
    df['smoking_status'] = le_smoking.transform(df['smoking_status'])
    
    with open(env.fullPath +  filePath + "scaler_bmi.pkl", 'rb') as file:
        scaler_bmi = pickle.load(file)
    df['bmi'] = scaler_bmi.transform(df[['bmi']])

    with open(env.fullPath +  filePath + "scaler_age.pkl", 'rb') as file:
        scaler_age = pickle.load(file)
    df['age'] = scaler_age.transform(df[['age']])

    prediction = loaded_model.predict(df)
    finalPredict = prediction[0].tolist()
    probability = loaded_model.predict_proba(df)

    confidenceN = probability[0][0]
    confidenceP = probability[0][1]

    print("==================================================")
    print("prediction : " + str(finalPredict))
    print("probability : " + str(probability))
    print("confidence of 0: " + str(confidenceN))
    print("confidence of 1: " + str(confidenceP))

    returnData = {
        "PREDICTION" : finalPredict,
        "CONFIDENT" : max([confidenceN, confidenceP]),
        "CONFIDENT_POSITIVE" : confidenceP,
        "CONFIDENT_NEGATIVE" : confidenceN,
    }

    return composeReply("SUCCESS", "Prediksi Stroke", returnData)


if __name__ == '__main__':
    app.run(host = env.runHost, port = env.runPort, debug = env.runDebug)