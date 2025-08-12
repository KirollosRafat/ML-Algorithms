import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as score
from time import time

## Customer Segmentation ##
# load data file
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

# Some data analysis
#print(cust_df.head()) 
#print(cust_df.tail())


# We will drop the address section as it's not relevant
cust_df = cust_df.drop('Address',axis=1)

# Check if there are any missing data
#print(cust_df.isnull().sum())

# seems like Defaulted Section has 150 NANs 
cust_df = cust_df.dropna() # let us just drop the nas
#print(cust_df.info())

# Check Correlation
correlation_values = cust_df.corr()['DebtIncomeRatio'].drop('DebtIncomeRatio')
#correlation_values.plot(kind='barh', figsize=(10, 6))
#plt.show()

# From correlation values it shows that 'edu', 'age', 'CustomerId' are no effect to the outcome
cust_df = cust_df.drop(['Customer Id'],axis=1)
#print(cust_df.info())

# Data Acquistion
# using iloc from pandas to allocate the features and the outcome
X = cust_df.iloc[:,[0,]]


# Splitting the data into training and testing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means model craetion
# k-means++ for better convergence, n_cluster is set to 3 as an initial guess, 12 repetitive 
start = time()
N_clusters = 3
k_means = KMeans(init="k-means++",n_clusters=N_clusters,n_init=12) 
k_means.fit(X_scaled)
end = time() - start

print(f"It took {end:.02f}secs to train the model")

# Create a Clus_km coulmn with the labels extracted
labels = k_means.labels_
cust_df["Clus_km"] = labels

print(f"{cust_df.groupby('Clus_km').mean()}\n")

silhouette_score = score(X,labels)
print(f"Silhouette Score = {silhouette_score:.02f}\n")

if silhouette_score >= 0.5:
    print("Approved Score ---> Good Clustering\n")
elif silhouette_score >= 0.25:
    print("Approved Score ---> Not Bad Clustering\n")
else:
    print("Not Approved ----> Weak Clustering\n")    