# file for creating model for house price
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
# reading and cleaning data
data = pd.read_csv("houseprice.csv")
# print(data.head())
# print(data.shape)
# print(data.describe())
# print(data.isnull().sum())
le = LabelEncoder()
sting_var = ['mainroad','guestroom','basement','hotwaterheating','airconditioning',
                                                                 'prefarea','furnishingstatus']
for i in sting_var:
    labels = le.fit_transform(data[i])
# print(labels)
    data.drop(i,axis=1,inplace=True)
    data[i] = labels
# print(data.furnishingstatus)
# print(data.ndim)
# print(data.mainroad.ndim)
variables = data.columns
x = data.drop('price',axis=1)
y = data['price']
print(x.ndim)
print(y.ndim)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

# creating a model pickle file
pickle.dump(model,open('houseprice.pkl','wb'))