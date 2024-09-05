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





