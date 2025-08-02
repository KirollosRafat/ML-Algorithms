
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as R2

# Load data
data = pd.read_csv("Salary_Data.csv")
exp_years = data['YearsExperience']
salary = data['Salary']

# Split into training and testing sets
years_train, years_test, salary_train, salary_test = train_test_split(exp_years, salary)

# Reshape features to 2D
years_train = np.array(years_train).reshape(-1, 1)
years_test = np.array(years_test).reshape(-1, 1)

# Train model
model = LinearRegression()
model.fit(years_train, salary_train)

# Predict
salary_predict = model.predict(years_test)

# Error 
R_squared = R2(salary_test,salary_predict)
print(f"R_squared Error = {R_squared:.02f}")

# Plot results
plt.figure(figsize=(8,6))
plt.scatter(years_train, salary_train, color='blue', label='Training data')
plt.plot(years_test, salary_predict, 'r', label='Predicted regression line')
plt.title("Prediction of Salaries According to The Years of Experience")
plt.xlabel("Experience Years")
plt.ylabel("Salary (USD)")
plt.legend()
plt.show()