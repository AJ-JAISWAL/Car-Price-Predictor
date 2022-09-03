# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 17:59:49 2022

@author: Anant Jaiswal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("CarPrice_Assignment.csv")

df.head()

df.shape

df.info()
df.duplicated().sum()
df.isnull().sum()

df.drop(columns=["car_ID","CarName"],inplace=True)
df.head()

sns.distplot(df['price'])

df['fueltype'].value_counts()
sns.barplot(x=df['fueltype'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

df['aspiration'].value_counts()
sns.barplot(x=df['aspiration'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

df.corr()['price']

df['symboling'].value_counts()

sns.barplot(x=df['doornumber'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(df['carlength'])
sns.scatterplot(x=df['carlength'],y=df['price'])

sns.distplot(df['carwidth'])
sns.scatterplot(x=df['carwidth'],y=df['price'])

df['stroke'].value_counts()

df['stroke']=df['stroke'].astype('int')
df['stroke']

def realstroke(a):
  if a == 3:
    return 4
  else:
    if a == 4:
      return 4
    else:
      return 2

df['real stroke']=df['stroke'].apply(realstroke)

df.drop(columns=["stroke",'symboling'],inplace=True)
df.head()

sns.barplot(x=df['real stroke'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

df.corr()['price']

df['drivewheel'].value_counts()
sns.barplot(x=df['drivewheel'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

df['carvolume']=df['carlength']*df['carwidth']*df['carheight']

df.corr()['price']

df['enginelocation'].value_counts()

df['cylindernumber'].value_counts()
sns.barplot(x=df['cylindernumber'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(df['curbweight'])

df['enginetype'].value_counts()
sns.barplot(x=df['enginetype'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

df['carbody'].value_counts()
sns.barplot(x=df['carbody'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(df['enginesize'])

df['fuelsystem'].value_counts()

sns.distplot(df['boreratio'])

df['mpg']=(df['citympg']+df['highwaympg'])/2

df.corr()['price']

df.info()

df.drop(columns=["citympg",'highwaympg','carlength','carwidth','carheight','compressionratio','peakrpm','enginelocation'
    ,'doornumber','cylindernumber','fuelsystem'],inplace=True)

df.info()

df.corr()['price']

y=df['price']
x=df.drop(columns=['price'])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)

"""step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,2,3,6])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
print('R2 score=',r2_score(y_test,y_pred))
print('MAE=',mean_absolute_error(y_test,y_pred))"""

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,2,3,6])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=10,
                              max_samples=0.9,
                              max_features=0.3,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
print('R2 score=',r2_score(y_test,y_pred))
print('MAE=',mean_absolute_error(y_test,y_pred))

'''from sklearn.linear_model import  BayesianRidge
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,2,3,6])
],remainder='passthrough')

step2 = BayesianRidge()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
print('R2 score=',r2_score(y_test,y_pred))
print('MAE=',mean_absolute_error(y_test,y_pred))'''

"""from sklearn.cross_decomposition import PLSRegression
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,2,3,6])
],remainder='passthrough')

step2 = PLSRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
print('R2 score=',r2_score(y_test,y_pred))
print('MAE=',mean_absolute_error(y_test,y_pred))"""

import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))









