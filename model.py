# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:25:53 2021

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


path = r'C:\Users\Dell\Documents\ml_material_proj'
df= pd.read_csv(os.path.join(path, r'data_eda.csv'))

# choose relevant columns

df_model = df[['Rating', 'Size', 'Type of ownership', 'Sector', 'Revenue', 'hourely', 'employer_provided', 
       'avg_salary', 'job_state', 'same_state','age', 'python', 'r_studio', 'spark', 'aws', 'excel', 'job_simplified',
       'seniority', 'desc_len', 'comp_counts']]

# get dummpy data

df_dum = pd.get_dummies(df_model)

# train test split 

from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum['avg_salary'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# multiple linear regression

import statsmodels.api as sm

X_sm = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
results.summary()

from sklearn.linear_model import LinearRegression, Lasso
import seaborn as sns

lm = LinearRegression()
lm.fit(X_train, y_train)
lm.score(X_test, y_test)
yhat_lm = lm.predict(X_test)

ax1 = sns.kdeplot(yhat_lm, color='b', label='Fitted values')
sns.kdeplot(y_test, color = 'r', label='Actual values', ax=ax1)

from sklearn.model_selection import cross_val_score, cross_val_predict

Rcross_lm = cross_val_score(lm, X_train, y_train, cv=4, scoring = 'neg_mean_absolute_error') # number of folds
Rcross_lm.mean()

yhat_lm_cv = cross_val_predict(lm, X_train, y_train, cv=4)

ax1 = sns.kdeplot(yhat_lm_cv, color='b', label='Fitted values')
sns.kdeplot(y_test, color = 'r', label='Actual values', ax=ax1)

# lasso regression

lm_lasso = Lasso()
lm_lasso.fit(X_train, y_train)
lm_lasso.score(X_test, y_test)

alpha = []
error = []

for i in range(1, 100):
    alpha.append(i /100)
    lm_lasso = Lasso(alpha = i / 100)
    lm_lasso.fit(X_train, y_train)
    error.append(np.mean(cross_val_score(lm_lasso, X_train, y_train, cv=3, scoring = 'neg_mean_absolute_error')))


plt.plot(alpha, error)

alpha[np.argmax(error)]

lm_lasso_mod = Lasso(alpha=0.17)
lm_lasso_mod.fit(X_train, y_train)
lm_lasso_mod.score(X_test, y_test)
yhat_lm_mod = lm_lasso_mod.predict(X_test)

ax1 = sns.kdeplot(yhat_lm_mod, color='b', label='Fitted values')
sns.kdeplot(y_test, color = 'r', label='Actual values', ax=ax1)

# random forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
yhat_rf = lm.predict(X_test)

ax1 = sns.kdeplot(yhat_rf, color='b', label='Fitted values')
sns.kdeplot(y_test, color = 'r', label='Actual values', ax=ax1)

Rcross_rf = cross_val_score(rf, X_train, y_train, cv=4, scoring = 'neg_mean_absolute_error')
Rcross_rf.mean()

# tune models GridsearchCV

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': range(10, 100, 10),'criterion': ('mse', 'mae'), 'max_features': ('auto', 'sqrt', 'log2')}
Grid = GridSearchCV(rf, parameters , cv = 3)
Grid.fit(X_train, y_train)
BestRR =Grid.best_estimator_
BestRR.score(X_test, y_test)

yhat_Grid = Grid.predict(X_train)

ax1 = sns.kdeplot(yhat_Grid, color='b', label='Fitted values')
sns.kdeplot(y_train, color = 'r', label='Actual values', ax=ax1)

Grid.best_estimator_

# test ensembels





