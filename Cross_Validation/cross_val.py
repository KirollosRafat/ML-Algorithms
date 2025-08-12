## Cross Validation 

# 1--> split the data into train and test
# 2 ---> The train data itself splitted into training set and validation set
# 3 ---> hyperparameter tunning to fint the best model

## Using k-fold cross validation 

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix

## load the iris data for practice
data = load_iris()
X, y = data.data, data.target
labels = data.target_names

## We will use pipelining  
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Standardize features
    ('pca', PCA(n_components=2),),       # Step 2: Reduce dimensions to 2 using PCA
    ('knn', KNeighborsClassifier(n_neighbors=5,))  # Step 3: K-Nearest Neighbors classifier
])

# Stratify the target when splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

## Pipeline integration
pipeline.fit(X_train, y_train)
# Measure the pipeline accuracy on the test data
test_score = pipeline.score(X_test, y_test)
print(f"Pipeline score : {test_score:.3f}\n") # score = 0.9

## Make predictions
y_pred = pipeline.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
# Create a single plot for the confusion matrix
plt.figure(figsize=(10,6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

# Set the title and labels
plt.title('Classification Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

## Tune hyperparameters using a pipeline within cross-validation grid search ##:

# Make a pipeline without specifying any parameters yet
pipeline = Pipeline([('scaler', StandardScaler()),
         ('pca', PCA()),
         ('knn', KNeighborsClassifier())])

# Hyper-parameter grid for tunning (dictionary like)
param_grid = {'pca__n_components': [2, 3], ## will use PCA with PC = 2 and PC = 3
              'knn__n_neighbors': [3, 5, 7] ## K can be 3,5,7
             }

# Use K-fold method for cross validation
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # n_splits range [5:10]

# Let's see what the best model is
best_model = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          cv=CV,
                          scoring='accuracy',
                          verbose=2
                         )

best_model.fit(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print(f"Best Model score : {test_score:.3f}\n")

for key,value in best_model.best_params_.items(): ## best_model.best_params_ returns a dictionary
    print(f"best {key} = {value}")


y_pred = best_model.predict(X_test)

# Generate the confusion matrix for KNN
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a single plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

# Set the title and labels
plt.title('KNN Classification Testing Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()