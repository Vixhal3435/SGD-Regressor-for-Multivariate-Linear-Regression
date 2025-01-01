# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Define the Problem

Identify the dependent variable (Y) and independent variables (X1, X2, ..., Xn) in the dataset.

2.Prepare the Dataset

Import the dataset and preprocess it (handle missing values, normalize, or standardize data if necessary).

3.Split Data

Divide the data into training and testing sets (e.g., 80% training, 20% testing).

4.Initialize Model

Use the SGDRegressor from the sklearn library. Specify parameters like loss='squared_error' and penalty (if regularization is desired).

5.Train the Model

Train the SGD model on the training dataset.

6.Make Predictions

Predict target values for the test set.

7.Evaluate the Model

Use metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) to evaluate model performance.

8.Visualize (Optional)

Visualize predicted vs actual results if dimensions permit.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: vishal.v
RegisterNumber:  24900179
*/
 import numpy as np
 from sklearn.datasets import fetch_california_housing
 from sklearn.linear_model import SGDRegressor
 from sklearn.multioutput import MultiOutputRegressor
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import mean_squared_error
 from sklearn.preprocessing import StandardScaler
 data = fetch_california_housing()
 x= data.data[:,:3]
 y=np.column_stack((data.target,data.data[:,6]))
 x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, 
random_state=0)
 scaler_x = StandardScaler()
 scaler_y = StandardScaler()
 x_train = scaler_x.fit_transform(x_train)
 x_test = scaler_x.fit_transform(x_test)
 y_train = scaler_y.fit_transform(y_train)
 y_test = scaler_y.fit_transform(y_test)
 sgd = SGDRegressor(max_iter=1000, tol = 1e-3)
 multi_output_sgd= MultiOutputRegressor(sgd)
 multi_output_sgd.fit(x_train, y_train)
 y_pred =multi_output_sgd.predict(x_test)
 y_pred = scaler_y.inverse_transform(y_pred)
 y_test = scaler_y.inverse_transform(y_test)
 print(y_pred)
 mse = mean_squared_error(y_test,y_pred)
 print("Mean Squared Error:",mse)
 print("\nPredictions:\n",y_pred[:5])
```
## Output:
![image](https://github.com/user-attachments/assets/0f059952-5a85-4c78-9a8a-328c8636d7a7)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
