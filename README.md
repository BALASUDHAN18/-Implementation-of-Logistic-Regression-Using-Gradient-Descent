# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required
  modules from sklearn.

## Program:
```
/*

Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: BALASUDHAN P
RegisterNumber:  212222240017

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data= np.loadtxt("/content/ex2data1 (2).txt", delimiter=',')
X= data[:, [0,1]]
y= data[:, 2]

print("Array value of X:")
X[:5]

print("Array value of Y:")
y[:5]

print("Exam 1-score graph:")
plt.figure()
plt.scatter(X[y==1][:, 0],X[y==1][:, 1], label="Admitted")
plt.scatter(X[y==0][:, 0],X[y==0][:, 1], label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print("Sigmoid function graph:")
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return J,grad
  
print("X_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

print("Y_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y) / X.shape[0]
  return grad 
  
print("Print res.x:")
X_train = np.hstack((np.ones((X.shape[0], 1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y), method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max= X[:,0].min()-1, X[:,0].max()+1
  y_min, y_max= X[:,0].min()-1, X[:,0].max()+1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted") 
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,y)

print("Probability value:")
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
  
print("Prediction value of mean:")
np.mean(predict(res.x,X) == y)  
*/
```

## Output:
![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/56f8a77b-8f55-4051-aeef-55a65ee9e646)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/92b8c20e-a133-4f51-8b6c-b40921efd473)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/df764f8c-54f5-483b-a938-6e4acc299295)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/fb736eec-d32c-4bd4-a151-802343c1373c)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/d6e3c25f-7842-4c62-9a7f-5a87c177120a)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/ad025dee-0ad2-4a1d-a6f8-b5da68c90dee)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/8b78b6ff-3572-4940-965d-3368327bc1a1)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/ca7321a4-f859-4e87-b3ac-f5183bf71ee1)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/4641afe2-cecb-4dce-a9b1-ea8eb82b1ddf)

![image](https://github.com/BALASUDHAN18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118807740/d22e96b4-4c39-464c-8f94-4abc2208ba87)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

