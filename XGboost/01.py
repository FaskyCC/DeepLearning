import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset_train = './data/train.csv'
dataset_test  = './data/test.csv'

data_train = pd.read_csv(dataset_train)
data_test = pd.read_csv(dataset_test)

x = data_train.drop(['ID', 'medv'], axis=1)
y = data_train.medv

X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bylevel=0.3, learning_rate=0.1, max_depth=8,
                           reg_alpha=8, n_estimators=500, reg_lambda=1)

xg_reg.fit(X_train, y_train)

x_test = data_test.drop(['ID'], axis=1)

predictions=xg_reg.predict(x_test)

ID = (data_test.ID).astype(int)
result = np.c_[ID, predictions]

np.savetxt('xgb_regression.csv', result, fmt='%d, %.4f', header='ID,medv', delimiter=',', comments='')