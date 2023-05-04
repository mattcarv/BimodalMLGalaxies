import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/mdocarm/Downloads/COMBO17.csv')

# Drop null rows
df = df.dropna()
df.shape

# Drop error columns

error = []

for f in df.columns:
    if f[0] == 'e':
        error.append(f)
        
df = df.drop(error, axis=1)

# Create a correlation matrix

new_df = df.drop('Nr', axis=1)

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
fig, ax = plt.subplots(figsize=(25, 18))
sns.heatmap(corr, annot=True, cmap='Blues', linewidth=0.5, mask=mask)
plt.title('Correlation matrix')
plt.clf()

plt.figure(figsize=(12, 10))
sns.scatterplot(new_df.Mcz, new_df.ApDRmag, label='Galaxy Size')
sns.scatterplot(new_df.Mcz, new_df.chi2red, alpha=0.5, label='chi2red')
plt.clf()
# chi2red > 5 is an outlier
# galaxy size below -2.5 is an outlier

df = df[new_df.chi2red<5]
df = df[new_df.ApDRmag>-2.5]

plt.figure(figsize=(12, 10))
sns.scatterplot(df.chi2red, df.ApDRmag, label='Galaxy Size')
sns.scatterplot(df.chi2red, df.mumax, alpha=0.5, label='Center of Galaxy')
plt.clf()

df = df[df.mumax>=20]
df = df[df.BjMAG<0]

plt.figure(figsize=(12, 10))
ax = sns.scatterplot(df.Mcz, df.BjMAG, label='Blue', alpha=0.5)
ax = sns.scatterplot(df.Mcz, df.rsMAG, label='Red', alpha=0.5)
plt.legend()
ax.invert_yaxis()
plt.clf()


df = df[df.Mcz<0.8]
df = df[df.Mcz>0.7]

plt.figure(figsize=(12, 10))
ax = sns.scatterplot(df.BbMAG, (df.UbMAG-df.BbMAG))
plt.title('Blue Mag vs Ultra-Blue')
plt.clf()

# reduce the dimensionality of the data

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(df)

print("Explained variance ratio: ")
print(pca.explained_variance_ratio_)

print('PCA components: ')
print(pca.components_)


df_pca = pca.transform(df)

# test the bimodality here

# df_bi = pd.DataFrame().assign(blue=df['BbMAG'], ultramb=(df['UbMAG']-df['BbMAG']))

# from sklearn.cluster import SpectralClustering

# # sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', 
# #                         assign_labels='kmeans', n_init=100)
# # label = sc.fit_predict(df_bi)

plt.scatter(df_pca[:,0], df_pca[:,1], s=15, linewidth=0)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# from sklearn import metrics

# # print(f"Silhouette Coefficient: {metrics.silhouette_score(df_bi, label):.3f}")

# from sklearn.cluster import KMeans

# km = KMeans(n_clusters=2, init='random', max_iter=1000)
# label = km.fit_predict(df_bi)

# plt.scatter(df_bi['blue'], df_bi['ultramb'], s=15, linewidth=0, c=label)
# plt.show()

# print(f"Silhouette Coefficient: {metrics.silhouette_score(df_bi, label):.3f}")
