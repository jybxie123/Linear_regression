import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rootPath = 'E:\working\LinearRegression\P46-Section4-Simple-Linear-Regression\Section 4 - Simple Linear Regression'
dataset = pd.read_csv(rootPath+'\Salary_Data.csv')
# print(dataset)

# from dataframe to ndarray
x = dataset.loc[:,['YearsExperience']].values
y = dataset.loc[:,['Salary']].values
# x = dataset.filter(items=['YearsExperience']).values
# x = dataset.iloc[:,:-1].values

# model selection
# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/4, random_state=0)

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# solution 1
# this way is too easy for us to get a linear regression, we need to calculate it by our selves
regressor.fit(x_train, y_train)

#predicting the test set results
y_pred = regressor.predict(x_test)


# visualising the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# visualising the test set results
plt.scatter(x_test, y_test, color='red')
# here we use any of x_train or x_test, it's ok, because it represents a straight line
plt.plot(x_test, regressor.predict(x_test), color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

