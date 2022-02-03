from matplotlib.pyplot import clf
import numpy as np
from sklearn.naive_bayes import GaussianNB

x= np.array([[-1,-1] , [-2,-1] , [-3,-2] , [1,1] , [2,1] ,[3,2]])
y= np.array([1,1,1,2,2,2])

clf = GaussianNB()
clf.fit(x,y)

print(clf.predict([[-0.8 , -1]]))
print(clf.predict([[1.2 , 1]]))