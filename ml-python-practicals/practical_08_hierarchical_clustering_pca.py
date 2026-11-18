import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

data = {'Age': [22, 25, 23, 45, 46, 48, 30, 32],
        'Income': [25000, 28000, 26000, 65000, 68000, 70000, 40000, 42000]}
df = pd.DataFrame(data)

# Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=2)
df['Cluster'] = hc.fit_predict(df[['Age', 'Income']])
print("Hierarchical Clustering Result:")
print(df)

# PCA
pca = PCA(n_components=1)
pc = pca.fit_transform(df[['Age', 'Income']])
print("Principal Component:")
print(pc.round(2))
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
