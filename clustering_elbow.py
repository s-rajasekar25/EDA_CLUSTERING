# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

# Step 1: Load the Dataset
df = pd.read_csv("C:\\Users\\sraja\\Downloads\\emailfile\\Mall_Customers.csv")
print(df.info())
print(df.describe())
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicated entries:\n{df.duplicated().sum()}")

# Step 2: Data Preprocessing
# Correct column names based on your dataset
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()  # Standardize data
scaled_features = scaler.fit_transform(features)

# Step 3: Determine Optimal Number of Clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot WCSS (Elbow Method)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 4: Apply K-Means Clustering
optimal_clusters = 3  # Set this based on the elbow point
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)
print(df.head())

# Step 5: Visualization
# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)
reduced_df = pd.DataFrame(reduced_features, columns=['PCA1', 'PCA2'])
reduced_df['Cluster'] = df['Cluster']

# 2D Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=reduced_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segments')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# successfully executed
