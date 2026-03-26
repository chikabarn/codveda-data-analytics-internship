import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_csv("cleaned_house_data.csv")

print(df.head())

# Select features for clustering
X = df.drop("MEDV", axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("elbow_method.png")
plt.close()

print("Elbow chart saved!")

# Apply KMeans with the optimal number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters


plt.scatter(df["RM"], df["MEDV"], c=df["Cluster"])

plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.title("K-Means Clustering of Houses")

plt.savefig("house_clusters.png")
plt.close()

print("Cluster plot saved!")