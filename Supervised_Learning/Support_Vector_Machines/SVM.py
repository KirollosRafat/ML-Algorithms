import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC # Support Vector Classifier


# read the input data
raw_data = pd.read_csv("diabetes.csv")
#print(raw_data.head(10))
#print(raw_data.tail(10))
#print(raw_data.info())

# Correlations for feature selection
correlation_values = raw_data.corr()['Outcome'].drop('Outcome')
correlation_values.plot(kind='barh', figsize=(10, 6))
plt.show()

# From data analysis we see that Skin Thickness and Blood Pressure have the lowest effect on outcome prediction
modified_data = raw_data.drop(['SkinThickness','BloodPressure'],axis=1)


# Feature Selection
X = modified_data.iloc[: ,[1,5]]
Y = modified_data.iloc[: , 6]

## Splitting
X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=30,test_size=0.25) 

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

## Creation Model and Performance
svm_model = LinearSVC()
svm_model.fit(X_train_scaled,y_train)

## Make predictions
y_predict = svm_model.predict(X_test_scaled)
Accuracy = accuracy_score(y_test,y_predict)
print(f"Model Accuracy = {Accuracy:.02f}\n") ## 0.77 when we dismissed two features

# Plotting confusion matrix
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True,fmt='d')
plt.show()

## It is better to select Descision Trees for this application with apporpraite depth but I used SVM for the sake of learning