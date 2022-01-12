from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

irisData = load_iris()
x = irisData.data
y = irisData.target
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2)

print('x_train :' , x_train.shape)
print('x_test : ' , x_test.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train , y_train)

print(knn.predict(x_test))