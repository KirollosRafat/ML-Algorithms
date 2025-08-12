import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor # Note to be taken We chose Regressor instead of Classifier

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)

# extract the labels from the dataframe
raw_data = raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)
y = raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

## Splitting
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

## Create the model and Train it
RegTree = DecisionTreeRegressor(criterion='squared_error',max_depth=5,random_state=42)
RegTree.fit(X_train,y_train)

## Make predictions 
y_predict = RegTree.predict(X_test)

MSE = mean_squared_error(y_test,y_predict)
print('MSE score : {0:.3f}'.format(MSE))
Model_r2_score = r2_score(y_test,y_predict)
print(f"Model R^2 score = {Model_r2_score:.02f}\n")

