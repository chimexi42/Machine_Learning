from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd



breast_cancer = load_breast_cancer()
x = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
print(x.head())
x = x[['mean area', 'mean compactness']]
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
y = pd.get_dummies(y, drop_first=True)


print(breast_cancer.feature_names)
print(breast_cancer.target)


print(y)
print(x)



x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print('Actual Y:', y_test)
print('predicted Y:',y_pred )




