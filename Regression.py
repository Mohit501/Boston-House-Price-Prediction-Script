from sklearn.linear_model import LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing  import StandardScaler
import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings(action='ignore')

def linearregression(x,y):
    model = LinearRegression()
    model.fit(x,y)
    return model

def ridge(x,y,alpha = 0.01):
    model = Ridge(alpha=alpha)
    model.fit(x,y)
    return model

def elasticnet(x,y,alpha = 0.01):
    model = ElasticNet(alpha = alpha)
    model.fit(x,y)
    return model

def lasso(x,y,alpha = 0.01):
    model = Lasso(alpha= alpha)
    model.fit(x,y)
    return model

def decisiontreeregressor(x,y):
    model = DecisionTreeRegressor(max_depth=6,min_samples_leaf=4,min_samples_split=3)
    model.fit(x,y)
    return model

def randomforestregressor(x,y):
    model = RandomForestRegressor(max_depth=4,min_samples_split=3,min_samples_leaf=3,n_estimators=100)
    model.fit(x,y)
    return model

def adadoostregressor(x,y):
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6,min_samples_leaf=3,min_samples_split=3),n_estimators=1000)
    model.fit(x,y)
    return model


#Loading Data
data = pd.read_csv("housing.data",delim_whitespace = True,header = None)
columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data.columns = columns

for i in data.columns:
    if i!='MEDV':
        data[i] = data[i].values.reshape(-1,1)
    else:
        break
x = data[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
y = data['MEDV']

#PreProcessing Data
sc1 = StandardScaler()
sc2 = StandardScaler()

x['B'] = sc1.fit_transform(x['B'].values.reshape(-1,1))
x['TAX'] = sc2.fit_transform(x['TAX'].values.reshape(-1,1))




#Splitting dataset into train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)

# Creating models
model1 = linearregression(x_train,y_train)
model2 = ridge(x_train,y_train)
model3 = elasticnet(x_train,y_train)
model4 = decisiontreeregressor(x_train,y_train)
model5 = randomforestregressor(x_train,y_train)
model6 = adadoostregressor(x_train,y_train)

#performance of models
print("Model Performance Paramaeters:")
print("")
print("1. Linear Regression:")
print("Insample Score:",model1.score(x_test,y_test))
print("2. Ridge Regression:",model2.score(x_test,y_test))
print("3. Elastic Net Regression:",model3.score(x_test,y_test))
print("4. Decision Tree Regressor:",model4.score(x_test,y_test))
print("5 Random Forest Regressor",model5.score(x_test,y_test))
print("6 AdaBoost Regressor",model6.score(x_test,y_test))

model = adadoostregressor(x,y)
with open('model_pkl','wb') as files:
    pickle.dump(model,files)


