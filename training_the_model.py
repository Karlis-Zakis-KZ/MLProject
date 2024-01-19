import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def normalize_time(t):
    try:
        # Convert the time to the standard datetime format
        return pd.to_datetime(t).time()
    except ValueError:
        # If the time cannot be converted, return a default value
        return pd.to_datetime('00:00:00').time()

# Load the dataset
df = pd.read_csv('ddos_data.csv')

# Fill missing values
df = df.fillna(0)

# Convert non-numeric data to numeric
df['timestamp'] = df['timestamp'].apply(normalize_time)

df['timestamp'] = df['timestamp'].apply(lambda t: pd.Timestamp.combine(pd.to_datetime('1970-01-01'), t)).astype('int64') / 10**9
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
for i, center in enumerate(reduced_centers):
    plt.scatter(center[0], center[1], marker='${}$'.format(i), s=150, c='red')

# Show the plot
plt.show()
# Print the cluster centroids
print(kmeans.cluster_centers_)

# Print the number of data points in each cluster
print(df['cluster'].value_counts())

# Optionally, save the DataFrame with cluster labels to a new CSV file
df.to_csv('ddos_data_with_clusters.csv', index=False)
