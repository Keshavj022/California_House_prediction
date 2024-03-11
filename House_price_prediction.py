#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn import linear_model

#fetching the datset
housing = fetch_california_housing(as_frame=True)
#printing essential details about the dataset
print(housing.data.shape)
print(housing.target.shape)
print(housing.feature_names)
print(housing)
#converting the dataset to a pandas dataframe
housing_dataframe = pd.DataFrame(housing.data)

housing_dataframe.head()
print(housing.target)

#checking missing values
housing_dataframe.isnull().sum()

#adding target variable to the dataframe
housing_dataframe['PRICE'] = housing.target
housing_dataframe.head()
#describing the dataset
housing_dataframe.describe()
correlation = housing_dataframe.corr()
print(correlation)
#plotting the correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar = True, square =True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap ='Blues')

X = housing_dataframe.drop(['PRICE'], axis=1)
Y = housing_dataframe['PRICE']

print(X)
print(Y)
scaler = StandardScaler()
#splitting training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state=2)
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
#making instance of the linear model
housing_model = linear_model.LinearRegression()

housing_model.fit(X_train, Y_train)
print("coeff = ", housing_model.coef_)
print("intercept = ",housing_model.intercept_)
#prediction on training data
training_data_prediction = housing_model.predict(X_train)
print(training_data_prediction)
#R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)
# mean absolute error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
print("R squared error = ", score_1)
print("Mean absolute error = ", score_2)

#prediction on test data
test_data_prediction = housing_model.predict(X_test)
print(test_data_prediction)
#R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)
# mean absolute error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("R squared error = ", score_1)
print("Mean absolute error = ", score_2)

#visualising the prediction
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual House Price VS Predicted House Price")
plt.show()
#checking house prices based on median income
scaler = StandardScaler()
a = housing_dataframe.iloc[:, :-1].values
b = housing_dataframe.iloc[:, [-1]].values

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 0)
#standardazing the dataframe
a_train = scaler.fit_transform(a_train)
a_test = scaler.transform(a_test)
b_train = scaler.fit_transform(b_train)
b_test = scaler.transform(b_test)

a_train_median_income = a_train[: , [7]]
a_test_median_income = a_test[: , [7]]

reg = linear_model.LinearRegression()

reg.fit(a_train_median_income, b_train)
predictionreg = reg.predict(a_test_median_income)

plt.scatter(a_train_median_income, b_train, color = 'blue')
plt.plot (a_train_median_income,
          reg.predict(a_train_median_income), color = 'red')
plt.title ('compare Training result - median income / median house value')
plt.xlabel('median income')
plt.ylabel('median house value')
plt.show()

plt.scatter(a_test_median_income, b_test, color = 'blue')
plt.plot (a_test_median_income,
          reg.predict(a_test_median_income), color = 'red')
plt.title ('compare Test result - median income / median house value')
plt.xlabel('median income')
plt.ylabel('median house value')
plt.show()

