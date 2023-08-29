# Linear_regression
An easy exercise of hand-written linear regression.
We use least square method and gradient descent method to solve the problems.

**As the dataset is small, we don't use validation set.**
### Least square method
train set
![image](https://github.com/jybxie123/Linear_regression/assets/66007115/e2eae20a-2c55-4553-b04e-1d24c16ad187)

test set
![image](https://github.com/jybxie123/Linear_regression/assets/66007115/d5d45485-5430-41d0-a5b0-bc4546cab397)


### Gradient descent method
We use Mini-batch GD to implement.
The loss goes like:
![image](https://github.com/jybxie123/Linear_regression/assets/66007115/fab8b02b-faf3-4dcc-a78e-3c13c700cc63)
The last epoch:
train set
![image](https://github.com/jybxie123/Linear_regression/assets/66007115/1289c5af-e556-4cf2-8883-ddb2f4de12d0)

test set
![image](https://github.com/jybxie123/Linear_regression/assets/66007115/f09fad2a-efa9-4ebf-9859-b14607f53a37)

There are some references for me to choose the mini-batch gradient descent.
https://zhuanlan.zhihu.com/p/72929546
