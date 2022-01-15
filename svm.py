import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris

irisData = load_iris()
x = irisData.data
y = irisData.target

clf = SVC(kernel='poly' , C=100 , gamma=1 , degree=5) 

clf.fit(x,y)

print('pridicted label :' , clf.predict([[2.7 , 1 , 7.5 , 5]]))

