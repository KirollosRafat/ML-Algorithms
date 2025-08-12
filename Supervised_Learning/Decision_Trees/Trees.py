import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix,accuracy_score



path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
data = pd.read_csv(path)
print(data.info())

## 4 out of the 6 features of this dataset are categorical (objects), which will have to be converted into numerical ones to be used for modeling. 
# For this, we can make use of __LabelEncoder__ from the Scikit-Learn library in preprocessing

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex']) 
data['BP'] = label_encoder.fit_transform(data['BP']) 
data['Cholesterol'] = label_encoder.fit_transform(data['Cholesterol']) # for example High --> 0 and Normal --> 1 

# Check if there are any missing values 
#print(data.isnull().sum())

custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
data['Drug_num'] = data['Drug'].map(custom_map) ## map each item on the Drug row to 0 or 1 or 2 or 3 or 4

# Check Correlation
#print(data.drop('Drug',axis=1).corr()['Drug_num'])

## Data preparation 
y = data['Drug']
X = data.drop(['Drug','Drug_num'], axis=1)

## Splitting the dataset into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

## Creation of the Model
# We chose Max depth to be 4 as we have five drugs if the accuarcy low increase the depth to 5
# Criteron is information gain based with a max depth = 4

DrugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
DrugTree.fit(X_train,y_train)

## Make Predictions and Evaluate Performance
Drug_predict = DrugTree.predict(X_test)
print(f"Desicion Tree Accuracy = {accuracy_score(y_test,Drug_predict):.02f}\n")

## Create Confusion Matrix
cm = confusion_matrix(y_test,Drug_predict)
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("Confusion Matrix of Decision Tree Model")
plt.show()

# Visulaization of the tree
plot_tree(DrugTree)
plt.show()