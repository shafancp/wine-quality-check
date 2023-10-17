from IPython.display import IFrame
IFrame('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', width=10, height=20)

# import load_iris function from datasets module
from sklearn.datasets import load_wine

# save "bunch" object containing iris dataset and its attributes
wine = load_wine()

# store feature matrix in "X"
X = wine.data

# store response vector in "y"
y = wine.target

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
knn.predict([[10,11,9,3,3,5,4,5,2,5,3,6,5]])