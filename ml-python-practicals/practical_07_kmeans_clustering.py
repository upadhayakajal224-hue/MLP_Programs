import pandas as pd
from sklearn.cluster import KMeans

data = {'Age': [22, 25, 23, 45, 46, 48, 30, 32],
        'Income': [25000, 28000, 26000, 65000, 68000, 70000, 40000, 42000]}
df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(df)
df['Cluster'] = kmeans.labels_

print("Clustered Data:")
print(df)
print("Cluster Centers:")
print(kmeans.cluster_centers_)
