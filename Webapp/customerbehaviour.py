import numpy as np
import pickle
from flask import Flask, request, render_template

model = pickle.load(open(
    'D:/Github/Customer Behaviour Analysis/webapp-pickle-dtm', 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    print(array_features)
    # Predict features
    prediction = model.predict(array_features)
    pred = "{:.2f}".format(*prediction)
    # pred = str(prediction)
    print(pred)

    return render_template('home.html', result=pred)


if __name__ == '__main__':
    app.run()
