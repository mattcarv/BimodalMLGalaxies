import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/mdocarm/Downloads/COMBO17.csv')

# Drop null rows
df = df.dropna()

# Drop error columns

error = []

for f in df.columns:
    if f[0] == 'e':
        error.append(f)
        
df = df.drop(error, axis=1)
df = df.drop('Nr', axis=1)

# Removing outliers

# sns.boxenplot(df.Rmag)
# sns.boxenplot(df.chi2red)
# sns.boxenplot(df.ApDRmag)
# sns.boxenplot(df.Mcz)


df = df[df.Rmag>19]
df = df[df.chi2red<5]
df = df[df.ApDRmag>-2.5]
df = df[df.Mcz<1.2]

df = pd.DataFrame().assign(Rmag=df['Rmag'], ApDRmag=df['ApDRmag'], Mcz=df['Mcz'],
                           chi2red=df['chi2red'])


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
minmax = MinMaxScaler()

df_scaler = scaler.fit_transform(df)
df_minmax = minmax.fit_transform(df)


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as shc


def agglomerative_cluster(dataset):
  silhouette_coefficient = []   #list to silhouette score
  for linkage in ['ward', 'complete', 'average', 'single']:   # the distance measuremtn will be based in
    for number in range(2,11):
      agglomerative = AgglomerativeClustering(n_clusters= number, affinity='euclidean', linkage = linkage) #agglomerative model
      agglomerative.fit(dataset)    #fitting the model
      score = silhouette_score(dataset, agglomerative.labels_)
      silhouette_coefficient.append(score)

  print(f'ward: {max(silhouette_coefficient[:9])}, {len(silhouette_coefficient[:9])}')
  print(f'maximum: {max(silhouette_coefficient[9:18])}, {len(silhouette_coefficient[9:18])}')
  print(f'average: {max(silhouette_coefficient[18:27])}, {len(silhouette_coefficient[18:27])}')
  print(f'minimum: {max(silhouette_coefficient[27:36])}, {len(silhouette_coefficient[27:36])}')

# agglomerative_cluster(dataset=df)


dendogram = shc.dendrogram(shc.linkage(df, method='single'), p=20, 
                           truncate_mode='lastp')
plt.clf()

#------------------------------------------------

def inertia_kmeans(data):
    inertia = []
    for i in range (1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=100, max_iter=1000,
                        random_state=1)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    # print(inertia)
    return inertia

inertia = inertia_kmeans(df)

plt.plot(range(1, 11), inertia)
plt.xticks(range(1, 11))
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.clf()

#-----------------------------------------------
# testing the silhouette score for different number of clusters

def sil_kmeans(data):
    sil_coef = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=100, max_iter=1000,
                        random_state=1)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        sil_coef.append(score)
    return sil_coef

sil_kmeans = sil_kmeans(df)

plt.plot(range(2, 11), sil_kmeans[:9])
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.clf()

#-----------------------------------------------------
# Using DBSCAN to test the silhouette score for different 
# distances and number of clusters

def sil_dbscan(data):
    sil_coef_db = []
    for radius in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            dbscan = DBSCAN(eps=radius, min_samples=n)
            dbscan.fit(data)
            score = silhouette_score(data, dbscan.labels_)
            sil_coef_db.append(score)
    
    print(f'radius-0.4: {max(sil_coef_db[:9])}')
    print(f'radius-0.5: {max(sil_coef_db[9:18])}')
    print(f'radius-0.6: {max(sil_coef_db[18:27])}')
    print(f'radius-0.7: {max(sil_coef_db[27:36])}')
    print(f'radius-0.8: {max(sil_coef_db[36:45])}')
    print(f'radius-0.9: {max(sil_coef_db[45:54])}')


sil_dbscan = sil_dbscan(df)

# We find that 0.9 is the best radius
# Now we get the coefficient scores for different number of samples within 
# this radius

sil_dbscan_coef = []
for n in range (2, 15):
    dbscan = DBSCAN(eps=0.9, min_samples=n)
    dbscan.fit(df)
    score = silhouette_score(df, dbscan.labels_)
    sil_dbscan_coef.append(score)

plt.plot(range(2, 15), sil_dbscan_coef)
plt.xlabel('Number of samples')
plt.ylabel('Silhouette Score')
plt.show()

# That gives us eps = 1.0 and min_samples=13 ?
#-------------------------------------------------------
