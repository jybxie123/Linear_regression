import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

rootPath = 'E:\working\LinearRegression\P46-Section4-Simple-Linear-Regression\Section 4 - Simple Linear Regression'
dataset = pd.read_csv(rootPath+'\Salary_Data.csv')

# number of x' feature is one  
x = dataset.loc[:,['YearsExperience']].values
y = dataset.loc[:,['Salary']]

# splitting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=4/5, random_state=0)

# init feature 
n_feature = 1


'''
# predict the values of y_train
y_pre = w.dot(x.T)+b

# calculate the error
# MSE
error = 1/2*np.mean((y_pre-y_train)**2)

# least square method
delta E/ delta W = δ(AX-Y)^2/δA = 2 X^T X A - 2 X^T Y = 0
W = (X^T X)^(-1) X^T Y
'''

# solution1 we use least square method
# we need to add a feature into x to represent b

x_train_1 = np.concatenate((np.array([1]*x_train.shape[0]).reshape(x_train.shape), x_train), axis=1)
x_test_1 = np.concatenate((np.array([1]*x_test.shape[0]).reshape(x_test.shape), x_test), axis=1)
w_best = np.linalg.inv(x_train_1.T.dot(x_train_1)).dot(x_train_1.T).dot(y_train)
print(w_best)
y_train_pre = x_train_1.dot(w_best)
y_test_pre = x_test_1.dot(w_best)
err = np.mean((y_test_pre-y_test)**2)
print(f'error of least square method: {err}')

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, y_train_pre, color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# visualising the test set results
plt.scatter(x_test, y_test, color='red')
# here we use any of x_train or x_test, it's ok, because it represents a straight line
plt.plot(x_test, y_test_pre, color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# solution2 we use gradient descent
err_list = []
# super parameter
batch_num = 5 # small batch updated
# batch_num = x_train.shape[0] # all samples as a batch
epochs = 500
e = 0
lr = 0.01
# init w and b
w = np.random.normal(0,1,(n_feature+1,1)) # 1 is for b
import math
while e <= epochs:
    idx = random.sample(list(range(x_train.shape[0])), batch_num)
    x_train_1 = np.concatenate((np.array([1]*x_train.shape[0]).reshape(x_train.shape), x_train), axis=1)
    x_test_1 = np.concatenate((np.array([1]*x_test.shape[0]).reshape(x_test.shape), x_test), axis=1)
    x_train_chosen = x_train_1[idx]
    y_train_chosen = y_train.values[idx]
    # x_train_chosen = x_train_1 # all samples as a batch
    # y_train_chosen = y_train.values
    delta = -2*(x_train_chosen.T.dot(x_train_chosen).dot(w) - x_train_chosen.T.dot(y_train_chosen))
    w = w + lr*1/batch_num*delta
    y_test_pre = x_test_1.dot(w)
    mse = np.mean((y_test-y_test_pre)**2)
    err_list.append(mse)
    e = e+1
    lr = 1/(1+e*0.00003) * lr # if we use all samples, don't need learning rate decay.
errs =  pd.Series(err_list).astype(float)
errs.plot(kind='line', grid = True, label= 'S1', title='Error', color='red')
plt.axhline(float(err),color='gray')
plt.show()

y_train_pre = x_train_1.dot(w)
y_test_pre = x_test_1.dot(w)
err = np.mean((y_test_pre-y_test)**2)
print(f'error of gradient descent method: {err}')


plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, y_train_pre, color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# visualising the test set results
plt.scatter(x_test, y_test, color='red')
# here we use any of x_train or x_test, it's ok, because it represents a straight line
plt.plot(x_test, y_test_pre, color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

'''
采用了批量梯度下降、小批量梯度下降、最小二乘法三种方法计算线性回归函数
最小二乘法显然为损失函数极小值，达到可训练的最优解。
批量梯度下降可以趋近最优解。但在接近最小值时面临梯度消失的问题，基本不再下降。
    其loss损失（训练集）曲线是平滑的，原因是全样本的回归函数即目标问题，因此下降方向总是正确的。
小批量梯度下降也可以趋近最小解，且不同于批量梯度下降，loss损失曲线抖动剧烈。
    这是因为小样本并不能代表整体分布，样本越小，差异越大，抖动是因为梯度计算的样本与整体回归函数不完全一致导致的。
    该下降方法可以也可以趋近于最小解。后期如果不降低学习率，抖动明显，无法有效收敛，如果降低学习率，则能够较为接近最优解。

'''