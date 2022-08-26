import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('houseprice.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    init_features = [i for i in request.form.values()]
    all_features = ['area','bedrooms','bathrooms','stories','mainroad_yes','mainroad_no','guestroom_yes','guestroom_no','basement_yes','basement_no','hotwaterheating_yes','hotwaterheating_no','airconditioning_yes',
    'airconditioning_no','parking','prefarea_yes','prefarea_no','furnishingstatus_unfurnished','furnishingstatus_semi-furnished','furnishingstatus_furnished']
    input_data = [0] * len(all_features)
    if init_features[0].isnumeric is True:
        for i in range(len(init_features)):
            if i in [0,1,2,3,9]:
                input_data[i]=(int(init_features[i]))
            else:
                if i in [4,5,6,7,8,10]:
                    if init_features[i] == "Yes":
                        input_data[i]=1
                        input_data[i+1]=0
                    elif init_features[i] == "No":
                        input_data[i]=0
                        input_data[i+1]=1
                else:
                    if init_features[i] == "unfurnished":
                        input_data[i]=1
                        input_data[i+1]=0
                        input_data[i+2]=0
                    elif init_features[i] == "semi-furnished":
                        input_data[i]=0
                        input_data[i+1]=1
                        input_data[i+2]=0
                    if init_features[i] == "furnished":
                        input_data[i]=0
                        input_data[i+1]=0
                        input_data[i+2]=1
    prediction = model.predict(np.array([input_data]))
    output = round(prediction[0],2)
    prediction_text = f'The price of the house is {output}'
    
    
    return render_template('index.html', prediction_text = {prediction_text})

        
    # elif (init_features[0].isnumeric == False):
    #     return render_template('index.html', prediction_text = 'Invalid input for area')

    


if __name__ == '__main__':
    app.run(debug=True)
