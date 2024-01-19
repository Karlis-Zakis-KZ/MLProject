import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('ddos_data.csv')

# Fill missing values
df = df.fillna(0)

# Convert non-numeric data to numeric
df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') / 10**9
df['src_mac'] = df['src_mac'].apply(lambda x: int(x.replace(':', ''), 16))
df['dst_mac'] = df['dst_mac'].apply(lambda x: int(x.replace(':', ''), 16))
df['src_ip'] = df['src_ip'].apply(lambda x: int(''.join([f'{int(i):08b}' for i in x.split('.')]), 2) if isinstance(x, str) else x)
df['dst_ip'] = df['dst_ip'].apply(lambda x: int(''.join([f'{int(i):08b}' for i in x.split('.')]), 2) if isinstance(x, str) else x)
df['tcp_flags'] = df['tcp_flags'].apply(lambda x: int(x, 16) if isinstance(x, str) else int(x))

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=10, n_init=10)
kmeans.fit(df_scaled)

# Predict clusters
predictions = kmeans.predict(df_scaled)

# Assign cluster labels to the original DataFrame
df['cluster'] = kmeans.labels_

# Reduce the scaled data to two dimensions using PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df_scaled)

# Reduce the cluster centers to two dimensions
reduced_centers = pca.transform(kmeans.cluster_centers_)

# Create a scatter plot of the reduced data with semi-transparent points
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=predictions, alpha=0.5)

# Plot the reduced cluster centers
plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], marker='x', s=150, c='red')

# Show the plot
plt.show()

# Print the cluster centroids
print(kmeans.cluster_centers_)

# Print the number of data points in each cluster
print(df['cluster'].value_counts())

# Optionally, save the DataFrame with cluster labels to a new CSV file
df.to_csv('ddos_data_with_clusters.csv', index=False)
