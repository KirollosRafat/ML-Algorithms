import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_iris = load_iris()

X = data_iris.data
y = data_iris.target

# Standarization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA 
pca = PCA() # We can control how many PCs we need through n_components passed to PCA() ---> PCA(n_components= ....)
X_pca = pca.fit_transform(X)

# Plotting
colors = ['navy', 'turquoise', 'darkorange']
lw = 1
for color, i, target_name in zip(colors, [0, 1, 2], data_iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=50, ec='k',alpha=0.7, lw=lw,
                label=target_name)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization")
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

# Showing the PCs percenatges for more understanding of PCA nature
PCs_percent = 100 * pca.explained_variance_ratio_

for i in range(len(PCs_percent)):
    print(f"PC{i+1} percentage = {PCs_percent[i]:.02f}%\n")

# PC3 and PC4 components are low in percentage and PC1 is the highest


