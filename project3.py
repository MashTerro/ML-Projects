# imprting all required libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(init='k-means++')
import seaborn as sns
sns.set()
from google.colab import files
import io

data_to_load = files.upload()
data = pd.read_csv('countries_1.csv')

plt.scatter(data['Longitude'],data['Accidents'])
plt.title('Scatter Graph')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

x = data.iloc[:,1:3]  # giving indices for accidents and longitude t
k=int(input("Enter value of k :"))  # Accepting values for number of clusters from user
kmeans = KMeans(k)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x) 
identified_clusters

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.title("Final Output")
plt.xlabel("Longitude")
plt.ylabel("Accidents ")
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Accidents'],c=data_with_clusters['Clusters'],cmap='rainbow')

print("\n Location  of Centriods ")
print(kmeans.cluster_centers_)



'''
User defined

import pandas as pd
import numpy as np
import random as rd

from google.colab import files
import io

data_to_load = files.upload()
data = pd.read_csv('countries_1.csv')

X = data[["Accidents","Longitude"]]
#Visualise data points
plt.scatter(X["Longitude"],X["Accidents"],c='blue')
plt.xlabel('Longitude')
plt.ylabel('Accidents')
plt.show()
k = int(input("Enter the number of clusters : "))

Centroids = (X.sample(n=k))
diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Longitude"]-row_d["Longitude"])**2
            d2=(row_c["Accidents"]-row_d["Accidents"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(k):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Accidents","Longitude"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Accidents'] - Centroids['Accidents']).sum() + (Centroids_new['Longitude'] - Centroids['Longitude']).sum()
        #print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Accidents","Longitude"]]
    
for k in range(k):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Longitude"],data["Accidents"],cmap='rainbow')

print("Centroids are : ")
print(Centroids);
print("\nK mean scatter graph as follows : ")
plt.scatter(Centroids["Longitude"],Centroids["Accidents"],c='Black')
plt.xlabel('Longitude')
plt.ylabel('Accidents')
plt.show()




'''