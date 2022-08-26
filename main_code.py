# file for creating model for house price
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
# reading and cleaning data
data = pd.read_csv("houseprice.csv")
for i in list(data.columns):
    data[i].fillna(0, inplace=True)
# print(data)
for i in ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']:
    p = pd.get_dummies(data[i],prefix=i)
    # data.drop(i)
    # data.append(x)
    data=pd.concat([p,data],axis=1)
# print(data)
data_cleaned=data.drop(['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus'],axis=1)
print(data_cleaned)
# data = data.drop('mainroad')
x = data_cleaned.drop('price',axis=1)
y = data_cleaned['price']
print(x.ndim)
print(y.ndim)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

#creating a model pickle file
pickle.dump(model,open('houseprice.pkl','wb'))
