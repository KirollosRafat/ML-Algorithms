import numpy as np
import pandas as pd
from matplotlib import  pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # Split the datasets into training and testing
from sklearn.metrics import r2_score,mean_squared_error# Our cost function for simple linear regression

df = pd.read_csv("Salary_Data.csv")


years = df['YearsExperience']
salary = df['Salary']

# Modify the shape of array for preparation
exp_years = np.array(years).reshape(-1,1) 
salaries = np.array(salary).reshape(-1,1)

# Splitting the data
x_train,x_test,y_train,y_test = train_test_split(exp_years,salaries,test_size=0.4,random_state=30) # 60% taining and 40% testing

# Create and Train the model
linear_model = LinearRegression()
linear_model.fit(x_train,y_train)

# Make predictions and judge the model performance
tolerance = 0.01
y_predict = linear_model.predict(x_test)
years_exp_test = np.array([[8.5]])
salray_predict = linear_model.predict(years_exp_test)

R2 = r2_score(y_test,y_predict)
MSE = mean_squared_error(y_test,y_predict)

print(f"MSE = {MSE:.02f}\n")
if 1 - round(R2) <= tolerance:
    print(f"R_squared_score = {R2:.02f}\n")
    print("R_2 score Accepetable\n")
else:
    print(f"R_squared_score = {R2:.02f}\n")
    print(f"Not Accepetable R_score should approach to one\n")    

print(f"If years of experience = {str(years_exp_test.flatten()[0])} then the expected salary = {salray_predict.flatten()[0]:.02f}")
## Plottings
figure = plt.figure(figsize=(6,6))
plt.scatter(x_test, y_predict, color='blue', label='Test Data', marker='x')
plt.plot(x_test,y_predict,color= 'red',label= 'Linear Regression Model')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Salary vs Experience")
plt.legend()
plt.grid(True)
plt.show()