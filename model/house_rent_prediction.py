# -*- coding: utf-8 -*-
"""House_Rent_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18BVdi-CGcQARZxLPaXWkdMbHz2EjAHPs
"""

import numpy as np
import pandas as pd

df = pd.read_csv('/content/House_Rent_Dataset.csv')

df.head()

df.corr()['Rent']

data = df.copy()

data.drop(columns=['Tenant Preferred', 'Posted On', 'Point of Contact', 'Area Type', 'Point of Contact', 'Floor', 'Area Locality'], inplace=True)

data.head()

data['City'] = data['City'].replace(["Mumbai","Bangalore","Hyderabad","Delhi","Chennai","Kolkata"],[5,4,3,2,1,0])
data['Furnishing Status'] = data['Furnishing Status'].replace(["Furnished","Semi-Furnished","Unfurnished"],[2,1,0])

data.corr()['Rent']

data.head()

X = data.drop('Rent', axis=1)
Y = data['Rent']

from sklearn.model_selection import train_test_split
X, x,Y,y = train_test_split(X,Y)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

models = {
    'ridge' : Ridge(),
    'xgboost' : XGBRegressor(),
    'lightgbm' : LGBMRegressor(),
    'gradient boosting' : GradientBoostingRegressor(),
    'lasso' : Lasso(),
    'random forest' : RandomForestRegressor(),
    'bayesian ridge' : BayesianRidge(),
    'support vector': SVR(),
    'knn' : KNeighborsRegressor(n_neighbors = 4)
}

for name, model in models.items():
    model.fit(X, Y)
    s = model.predict(x)
    score = r2_score(y,s)
    print(f'{name} trained')
    print(f'{score} score')

test = [[2	,	1100,	0,	0,	2]]
models['random forest'].predict(test)

import pickle

filename = 'predictor.pkl'
pickle.dump(model, open(filename, 'wb'))

