import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pandas as pd
import skdim
from scipy.stats import pearsonr

# Generate synthetic datasets

# Torus dataset
def generate_torus_point_cloud(num_points=5000, R=3, r=1):
    # Generate random angles for theta and phi
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    # Compute the torus points
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    x_c = R * np.cos(theta)
    y_c = R * np.sin(theta)
    z_c = np.zeros(x.shape)

    K = np.cos(phi) / (r * (R + r * np.cos(phi)))

    return np.column_stack((x, y, z)), np.column_stack((x_c, y_c, z_c)), K


# torus parameters

R = 1 # Major radius
r = 0.375  # Minor radius
num_samples = 5000

# Generate a torus point cloud with 1000 points and radius 1

torus, torus_centers, torus_K = generate_torus_point_cloud(num_points = num_samples, R = R, r = r)

torus += np.random.normal(0, 0.0, torus.shape)

# Visualize the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(torus[:, 0], torus[:, 1], torus[:, 2], s=1, c = torus_K)
#ax.scatter(torus[0, 0], torus[0, 1], torus[0, 2], s=10, c = 'r')
#ax.scatter(torus_centers[:, 0], torus_centers[:, 1], torus_centers[:, 2], s=5, c = 'r')
ax.set_aspect('equal')
ax.set_title("Generated torus Point Cloud")
ax.view_init(45, 0)
plt.show()

idx = 2000
query = torus[idx].reshape(1, -1)
k = int(0.2 * len(torus))
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(torus)
ep_dist, ep_idx = nbrs.kneighbors(query, k, return_distance=True)
ep_dist.shape

#
num_eval = int(len(torus))
gaussian_cur = []
mean_cur = []
for i in tqdm(range(num_eval)):
    g, m = compute_curvature_adaptive(torus, torus[i].reshape(1, -1))
    gaussian_cur.append(g)
    mean_cur.append(m)

v = np.array(gaussian_cur).T
# Visualize the point cloud

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(torus[:num_eval, 0], torus[:num_eval, 1], torus[:num_eval, 2], s=2, c = v)
ax.set_title("Gaussian curvature, ep_PCA = 0.2, tau is chosen according to the eigenvalue ratio plot")
ax.view_init(90, 0)
plt.colorbar(scatter)
ax.set_aspect('equal')
plt.show()


v = np.array(mean_cur).T
# Visualize the point cloud
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(torus[:num_eval, 0], torus[:num_eval, 1], torus[:num_eval, 2], s=2, c = -v)
ax.set_title("Curvature on point cloud, ep_PCA = 0.2, tau_radius = 1, max_min = 80")
ax.view_init(90, 0)
plt.colorbar(scatter)
ax.set_aspect('equal')
plt.show()


# Ellipsoid dataset

def generate_ellipsoid_cloud(a, b, c, num_points=5000):
    """Generate a random point on an ellipsoid defined by a,b,c"""

    theta = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.rand(num_points)
    phi = np.arccos(2.0 * v - 1.0)
    sinTheta = np.sin(theta);
    cosTheta = np.cos(theta);
    sinPhi = np.sin(phi);
    cosPhi = np.cos(phi);
    rx = a * sinPhi * cosTheta;
    ry = b * sinPhi * sinTheta;
    rz = c * cosPhi;
    K = 1 / (a ** 2 * b ** 2 * c ** 2 * (rx ** 2 / a ** 4 + ry ** 2 / b ** 4 + rz ** 2 / c ** 4) ** 2)
    return np.column_stack((rx, ry, rz)), K

ellipsoid, ellip_K = generate_ellipsoid_cloud(0.9, 1.5, 0.9)

# Visualize the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ellipsoid[:, 0], ellipsoid[:, 1], ellipsoid[:, 2], s=5, c = ellip_K)
ax.set_aspect('equal')
ax.set_title("Generated Sphere Point Cloud")
ax.view_init(0, 0)
plt.show()


num_eval = int(len(ellipsoid))
gaussian_cur = []
mean_cur = []
for i in tqdm(range(num_eval)):
    g, m = compute_curvature_adaptive(ellipsoid, ellipsoid.reshape(1, -1))
    gaussian_cur.append(g)
    mean_cur.append(m)

v = np.array(gaussian_cur).T
corr, _ = pearsonr(ellip_K , v)
corr
