
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as R2, mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as R2, mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv("Real estate.csv")

# Select features and target
features = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']]
y = df['Y house price of unit area']

# Polynomial feature expansion (degree 2 or 3)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(features)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
poly_mae = MAE(y_test, y_pred)
poly_mse = MSE(y_test, y_pred)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = R2(y_test, y_pred)

# Print metrics
print(pd.DataFrame(
    [poly_mae, poly_mse, poly_rmse, poly_r2],
    index=['MAE', 'MSE', 'RMSE', 'R²'],
    columns=['metrics']
))

# Plot: Predicted vs Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='teal', edgecolors='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual House Price per Unit Area")
plt.ylabel("Predicted House Price per Unit Area")
plt.title("Predicted vs Actual Values")
plt.grid(True)
plt.show()
