from sklearn.datasets import load_iris
from sklearn import tree

x,y = load_iris(return_X_y=True)

print('x.shape:' , x.shape)
print('y.shape:' , y.shape)
print('x.data:' , x[48:53])
print('x.data:' , x[48:53])
print('y.data:' , y[48:53])

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x,y)

print(clf.predict([[1.1,4.3,2.0,1.9]]))


