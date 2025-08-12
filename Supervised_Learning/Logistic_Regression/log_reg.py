import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import  pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix ## For model performance eval
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("User_Data.csv")

#print(df.describe())
#print(df.head())

## Select features of Age and Estimated salary
X = df.iloc[:,[2,3]] 
Y = df.iloc[:,4]

## Spli Dataset into training and testing
X_train, X_test, y_train,y_test = train_test_split(X,Y,random_state = 30,test_size=0.2,shuffle=True) # 20% percentage for testing

## Feature Scaling
scaler = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

## Classification Model
classifier = LogisticRegression()
classifier.fit(X_train_scaled,y_train)

## Make Predictions
y_predict = classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test,y_predict)
print(f"Model Accuracy Score = {accuracy:.02f}\n")

Cm = confusion_matrix(y_test,y_predict)

## Plotting the Confusion Matrix
sns.heatmap(Cm,annot=True,fmt='.2g')
plt.title("Purchased Item Confusion Matrix")
plt.xlabel("Predicted Purchasing")
plt.ylabel("True Purchasing")
plt.show()

# Plotting the Logistic Regression Decision Boundary

# Create a meshgrid for plotting decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict over the grid
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                      c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)

plt.xlabel('Age (scaled)')
plt.ylabel('Estimated Salary (scaled)')
plt.title('Logistic Regression Decision Boundary')
plt.legend(*scatter.legend_elements(), title="Purchased")
plt.show()