# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:07:09 2021

@author: Dell

This is just summary of diffrerent commands
"""

# some import 

import pandas as pd
import numpy as np


# replace A by B

df.replace(A, B, inplace = True)

# assume we have NaN how to make those NaN to True in df

df.isnull()
df.notnull()

# what to use after value_counts() method to see the most category in the df

df['something'].value_counts().idxmax()

# reset the index in df

df.reset_index(drop = True, inplace = True)

# to change a type of a df to int, float, object

df[['v1', 'v2']] = df[['v1', 'v2']].astype('int')

# to rename a columns name 



df.rename(columns = {'old_name' : 'new_name'}, inplace = True)

# define the horsepower column in three bins Low, Medium, High

bins = np.linspace(min(df.horsepower), max(df.horsepower), 4)
names = ['Low', 'Medium', 'High']

df['hosrepower_binned'] = pd.cut(df['horsepowr'], bins, labels = names, include_lowest = True)

# make a dummy df from fuel type column

df_fuel_type_dummy = pd.get_dummies(df['fuel_type'])

# how to concate two df 

df = pd.concat([df, df_fuel_type_dummy], axis = 1)

# to find correlation between all continous variables in 

df.corr()

# great way for visualization of correlation between continous varialble is scatterplot or regplot in seaborn

import seaborn as sns

sns.regplot(x='', y='', data=df)
plt.ylim(0,)

# categorical variables can have the type of the object or int
# a good way to visualize categorical variables is by boxplot in seaborn

sns.boxplot(x='', y='', data=df)

# in the describe method to see just the categorical variables 

df.describe(include = 'object')

# one bracket [] is called series and two brackets [[]] called dataframe
# value_counts() method work on series not dataframes

convert series to data frame

df[].value_counts().to_frame()

# to put name on the index of df

df.index.name = 'the name'

# to see which unique categorical variables are in a column

df[''].unique()

# use groupby method group and average 

df_group = df[['v1', 'v2', 'target v']]
df_group_test = df_group.groupby(['v1', 'v2'], as_index = False).mean()

# convert groupby df to pivot table

grouped_pivot = df_group_test(index='v1', columns='v2')

# to fill NaN with 0 in df

df = df.fillna(0)

# P-value < 0.001 statistically significant
# P-value < 0.05 statistically moderate
# P-value < 0.1 statistically weak
# P-value > 0.1 statistically no evidence which correlation is significant

# get P-value , it is needed to use stats library

from scipy import stats

pearson_coef, P-value = stats.pearsonnr(df['v1'], df['target v'])

# what is ANOVA, analysis of variance
# ANOVA is for categorical varial return two values F-test score and P-value
# F-test score assume all mean is the same, the bigger F-score the means are far from each other
# P-value statistically significanse 

f_val, p_val = stats.f_oneway(groupdf.get_group('fwd')['price'], groupdf.get_group('rwd')['price']) 

# linear regression

from sklearn.linear_model import LinearRegression

# draw residual plot in seaborn

sns.residplot(x='', y='', data = df)

# visualize multiple linear regression, distribution plot in seaborn

ax1 = sns.distplot(yhat, hist=False, color='b', label='Fitted values')
sns.distplot(df['price'], hist=False, color = 'r', label='Actual values', ax=ax1)

# distplot is deprecated alternative function is kdeplot

 ax1 = sns.kdeplot(yhat, color='b', label='Fitted values')
sns.kdeplot(df['price'], color = 'r', label='Actual values', ax=ax1)

# Polynominal regression for single independent variable
# it is a numpy method

f = np.polyfit(x, y, 3)
p = np.poly1d(f) # parameters

# Polynomial regression for multiple independent variable

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=5)
Z_pr = pr.fit_transform(Z)
poly = LinearRegression()
poly.fit(Z_pr, y)

# what is the other name for R, coeficient of determination

lm.score(x, y) # simple linear model R value if it is 0.49 than means 49% of data explaned with this model

# mean square error

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['target value'], yhat)

# to get just numerical values of df

df = df._get_numeric_data()

# cross validation 

from sklearn.model_selection import cross_val_score

Rcross = cross_val_score(lrmodel, x, y, cv=4) # number of folds
Rcross.mean()

from sklearn.model_selection import cross_val_predict

yhat = cross_val_predict(lrmodel, x, y, cv=4)

# if one would like to find in Polynominal regression which order and which dividing
# just give you possibility to check different posibilities 

pip install ipywidgets
from ipywidgets import interact

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower', ...]])
    x_test_pr = pr.fit_transform(x_test[['horsepower', ...]])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    print(order)
    print(test_data)
    
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))

# Ridge regression using alpha value

from sklearn.linear_model import Ridge

RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
yhat = RigeModel.predict(x_test_pr)

# grid search to find the best parameter for alpha and 

from sklearn.model_selection import GridSearchCV

parameters2= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]
RR = Ridge()
Grid1 = GridSearchCV(Ridge(), parameters2,cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR=Grid1.best_estimator_
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)












