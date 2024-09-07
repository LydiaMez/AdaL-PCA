import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pandas as pd
import pickle
import phate
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from numpy import savetxt
import plotly.graph_objs as go
import scprep
import skdim
import scanpy as sc


Y_phate = pd.read_csv('output/ebdata_2d.csv', sep=',', header=None)
Y_phate = np.array(Y_phate)

labels = pd.read_csv('output/ebdata_labels.csv', sep=',', header=None)
labels = np.array(labels)

Y_phate3d = pd.read_csv('output/ebdata_3d.csv', sep=',', header=None)
Y_phate3d = np.array(Y_phate3d)
eb_3d = scipy.stats.zscore(Y_phate3d)



def find_basis(point_cloud, x,  extrin_dim = 3, epsilon_PCA = 0.1, tau_radius = 0.4):
    #point_cloud: the manifold 
    #x: np.array of shape 1 by p, the point where the curvature is evaluated at, e.g., [[1, 2, 3]]
    #epsilon: the radius of local PCA
    #dim: the dimension of the manifold
    #tau_ratio: the ratio is tau radius (where we evaluate the curvature)/ epsilon_sqrt
    
    # Find transport neighborhood
    k = int(0.05 * point_cloud.shape[0])

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(point_cloud)
    ep_dist, ep_idx = nbrs.radius_neighbors(x, epsilon_PCA, return_distance=True, sort_results = True)
    
    tau_dist, tau_idx = nbrs.radius_neighbors(x, tau_radius, return_distance=True, sort_results = True)
    
    tau_nbrs = point_cloud[tau_idx[0]]
    
    
    pca_nbrs = point_cloud[ep_idx[0]]
    Xi = pca_nbrs - x
    Di = np.diag(np.sqrt(np.exp(- np.array(ep_dist[0]) ** 2 / epsilon_PCA)))
    Bi = Xi.T @ Di
    
    U, S, VT = np.linalg.svd(Bi.T, full_matrices = False)
    O = VT[:extrin_dim, :]
    
    
    return tau_nbrs, O

def compute_Gaussian_curvature(point_cloud, query_point, extrin_dim = 3, 
                               epsilon_PCA = 0.1, tau_radius = 0.4, max_min_num = 10, use_cross = True):
    
    tau_nbrs, O = find_basis(point_cloud, query_point, extrin_dim = extrin_dim,
                             epsilon_PCA = epsilon_PCA, tau_radius = tau_radius)
            
    if use_cross:
        O2 = np.cross(O[0], O[1])
    else:
        O2 = O[2]

    ti = tau_nbrs[1:] - tau_nbrs[0]
    norms = np.square(ti).sum(axis=1)
    tensor_all = 2 * (O2 * ti).sum(axis=1) / norms

    if max_min_num < 1:
        min_quantile = max_min_num
        max_cur = np.quantile(tensor_all, 1-min_quantile)
        min_cur = np.quantile(tensor_all, min_quantile)
    else:
        max_cur = sum(sorted(tensor_all, reverse=True)[:max_min_num])/max_min_num    
        min_cur = sum(sorted(tensor_all)[:max_min_num])/max_min_num
    
    return max_cur * min_cur


def vector_projection(v, v1, v2):
    # Compute dot products
    dot_v_v1 = np.dot(v, v1)
    dot_v_v2 = np.dot(v, v2)

    # Compute the projection
    projection = dot_v_v1 * v1 + dot_v_v2 * v2
    projection = projection / np.linalg.norm(projection)

    return projection



def compute_principal_direction(point_cloud, query_point, extrin_dim = 3, 
                                epsilon_PCA = 0.1, tau_radius = 0.4, max_min_num = 10, use_cross = True):
    
    tau_nbrs, O = find_basis(point_cloud, query_point, extrin_dim = extrin_dim,
                             epsilon_PCA = epsilon_PCA, tau_radius = tau_radius)
            
    if use_cross:
        O2 = np.cross(O[0], O[1])
    else:
        O2 = O[2]
        
    max_min_num = int(0.3 * len(tau_nbrs))

    ti = tau_nbrs - query_point[0]
    norms = np.square(ti).sum(axis=1)
    tensor_all = 2 * (O2 * ti).sum(axis=1) / norms
    
    max_indices = np.argsort(tensor_all)[-max_min_num: ]
    principal_dir1 = tau_nbrs[max_indices] - query_point[0]
    chose_dir1 = principal_dir1[round(max_min_num/2)]
    dir1_idx = np.where(np.dot(principal_dir1, chose_dir1)>=0)[0]
    principal_dir1 = principal_dir1[dir1_idx]
    
    #prin_norms1 = np.square(principal_dir1).sum(axis=1)
    #principal_dir1 = principal_dir1 / prin_norms1[:, np.newaxis]
    principal_dir1 = principal_dir1.sum(axis=0)/len(principal_dir1)
    principal_dir1 = vector_projection(principal_dir1, O[0], O[1])
    
    min_indices = np.argsort(tensor_all)[:max_min_num]
    principal_dir2 = tau_nbrs[min_indices] - query_point[0] 
    chose_dir2 = principal_dir2[round(max_min_num/2)]
    dir2_idx = np.where(np.dot(principal_dir2, chose_dir2)>=0)[0]
    principal_dir2 = principal_dir2[dir2_idx]
    
    #prin_norms2 = np.square(principal_dir2).sum(axis=1)
    #principal_dir2 = principal_dir2 / prin_norms2[:, np.newaxis]
    principal_dir2 = principal_dir2.sum(axis=0)/len(principal_dir2)
    principal_dir2 = vector_projection(principal_dir2, O[0], O[1])
    
    return principal_dir1, principal_dir2
 
    
num_eval = int(len(eb_3d))

curvature = []
for i in tqdm(range(num_eval)):
    b = compute_Gaussian_curvature(eb_3d, eb_3d[i].reshape(1, -1), 
                                   epsilon_PCA =0.55, tau_radius = 1.5, max_min_num = 3000)
    curvature.append(b)
    
v = np.array(curvature).T


def plot_vector(ax, a, v, color='red', label=None):
    ax.quiver(a[0], a[1], v[0], v[2], color=color, label=label)
    
    
    
    
idx1 = np.where(v == min(v))[0][0]
idx1 = int(idx1 + 0)

idx2 = 52
a, b = compute_principal_direction(eb_3d, eb_3d[idx1].reshape(1, -1), extrin_dim = 3, 
                                   epsilon_PCA = 0.3, tau_radius = 1)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

scprep.plot.scatter2d(Y_phate, c=dd, figsize=(12,8), cmap=cmap, ax= ax,
                      ticks=False, label_prefix="PHATE", title = "Gaussian Curvature", fontsize = 10, s = 5)
def plot_vector(ax, a, v, color='red', label=None):
    ax.quiver(a[0], a[1], v[0], v[2], color=color, label=label)
plot_vector(ax, Y_phate[idx2], a)
plot_vector(ax, Y_phate[idx2], b)
def plot_vector(ax, a, v, color='red', label=None):
    ax.quiver(a[0], a[1], v[0], v[1], color=color, label=label)
    
plot_vector(ax, Y_phate[idx1], a)
plot_vector(ax, Y_phate[idx1], b)

