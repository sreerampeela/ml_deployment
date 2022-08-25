import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('houseprice.pkl','rb'))

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    x = []
    x.append([[request.form.get('area')]])
    x.append([[request.form.get('bedrooms')]])
    x.append([[request.form.get('bathrooms')]])
    x.append([[request.form.get('stories')]])
    x.append([[request.form.get('mainroad')]])
    x.append([[request.form.get('guestrooms')]])
    x.append([[request.form.get('basement')]])
    x.append([[request.form.get('hotwaterheating')]])
    x.append([[request.form.get('airconditioning')]])
    x.append([[request.form.get('parking')]])
    x.append([[request.form.get('prefarea')]])
    x.append([[request.form.get('furnishingstatus')]])
    x2=np.array(x)
    nfeatures,nx,ny = x2.shape
    x_new = x2.reshape((nfeatures,nx*ny))
    x_new = x_new.values.astype(np.float64)
    # x=np.concatenate(x)
    # x.reshape(-1,1)
    prediction = model.predict(x_new)
    output = round(prediction[0],2)
    return render_template('index.html', prediction_text = f'The price of the house is {output}')


if __name__ == '__main__':
    app.run(debug=True)