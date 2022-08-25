import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('houseprice.pkl','rb'))
x = []
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    
    x.append((float(request.form.get('area')), float(request.form.get('bedrooms')), float(request.form.get('bathrooms')), 
              float(request.form.get('stories')), float(request.form.get('mainroad')), float(request.form.get('guestrooms')),
              float(request.form.get('basement')),float(request.form.get('hotwaterheating')),float(request.form.get('airconditioning')),
              float(request.form.get('parking')),float(request.form.get('prefarea')),float(request.form.get('furnishingstatus'))))
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
