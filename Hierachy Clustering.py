import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

x = np.array([[5,3],
              [10,15],
              [15,12],
              [24,10],
              [30,30],
              [85,70],
              [71,80],
              [60,78],
              [70,55],
              [80,91], ])

labels = range(1,11)
plt.scatter(x[:, 0], x[:,0], label = 'True Position')

for label,x,y in zip(labels, x[:,0],x[:,1]):
    plt.annotate(label, xy=(x,y), xytext =(-3,3),
            textcoords= 'offset points', ha='right',va = 'bottom')

plt.show()