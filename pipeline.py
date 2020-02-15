#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
pd.options.display.max_rows = 400
pd.options.display.max_columns = 100
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt



house_data = pd.read_csv("train.csv")
house_data.head(5)


cat_var = list(house_data.select_dtypes(include=["O"]).columns)
num_var = [c for c in house_data.columns if c not in cat_var+['SalePrice']]
id_var_cols = [house_data.columns.get_loc(c) for c in cat_var if c in train_data]
id_num_cols = [house_data.columns.get_loc(c) for c in num_var if c in train_data]
cat_var



steps_num = [('imputer_num', SimpleImputer(strategy = 'median')), ('scaler', StandardScaler())]
steps_var = [('imputer_cat', SimpleImputer(strategy = 'most_frequent')),('encoder', OneHotEncoder(handle_unknown= 'ignore'))]

pipeline_num = Pipeline(steps_num)
pipeline_var = Pipeline(steps_var)

pre_processor = ColumnTransformer(transformers = [('trans_num',pipeline_num,id_num_cols),('trans_cat',pipeline_var,id_var_cols)])
steps_grid = [('preprocessor', pre_processor),('RFR', RandomForestRegressor())]

pipeline_grid = Pipeline(steps_grid)



Y = house_data[['SalePrice']].values
X = house_data.drop(['SalePrice'], axis = 1).values
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3,random_state = 20)



parameters = {'RFR__n_estimators':[50,100,150], 'RFR__max_depth':[None,1,2], 'RFR__min_samples_split':[2,3,4]}
grid = GridSearchCV(pipeline_grid, param_grid = parameters, cv=5)
grid.fit(xtrain,ytrain)
ypred = grid.predict(xtest)
RMSE = np.sqrt(mean_squared_error(ytest, ypred))
print ("R2 score = {}".format(grid.score(xtest,ytest)))
print ("RMSE score = {}".format(RMSE))
print (grid.best_params_)



x=ytest
y=ypred    # define your axis

plt.plot(x,y,'r.') # 
plt.plot(x,x,'k-') # line separator
plt.xlim(0,800000)
plt.ylim(0,800000)
plt.xlabel('Actual HousePrice - ytest')
plt.ylabel('Predicted HousePrice- ypred')
plt.show()

