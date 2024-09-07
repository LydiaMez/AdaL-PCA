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
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random






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
 
    
def make_ipsc(X, n_obs=150,emb_dim=3,knn=5,indx=None):
    
    #load data
    #initdir = os.getcwd()
    #os.chdir(os.path.abspath('..') + '/src/data')
    #X = sio.loadmat('ipscData.mat')['data']
    #os.chdir(initdir)
    ipsc_data = X[indx,:].squeeze()
    
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn,t=250,decay=10)
    ipsc_phate = phate_operator.fit_transform(ipsc_data) #only compute on 100 points since it's so expensive for > 6000
    ipsc_phate = scipy.stats.zscore(ipsc_phate) 
    
    return ipsc_phate

n_obs=10000
nontrain = list(np.arange(X.shape[0]))
testind = random.sample(nontrain,n_obs)


make_data = True
if make_data:
    ipsc = make_ipsc(X, n_obs=n_obs,emb_dim=3,knn=5,indx=testind)



num_eval = int(len(ipsc))
curvature = []
for i in tqdm(range(num_eval)):
    b = compute_Gaussian_curvature(ipsc, ipsc[i].reshape(1, -1), 
                                   epsilon_PCA =0.7, tau_radius = 1.5, max_min_num = 0.3)
    curvature.append(b)

v = np.array(curvature).T

def sample_nearest_k(X, query, k):
    nbrs =NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    idx = nbrs.kneighbors(query, k, return_distance=False)
    return X[idx][0], idx



def plot_vector(ax, a, v, color='red', label=None):
    ax.quiver(a[0], a[1], a[2], v[0], v[1], v[2], color=color, label=label)
    
idx1 = 100
idx2 = 2

a1, b1 = compute_principal_direction(ipsc, ipsc[idx1].reshape(1, -1), extrin_dim = 3, 
                                     epsilon_PCA = 0.3, tau_radius = 0.2, max_min_num = 0.3)
sub1, sub_idx1 = sample_nearest_k(ipsc, ipsc[idx1].reshape(1, -1), 19000)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')

# Scatter plot
scatter = ax.scatter(sub1[:,0], sub1[:,1], sub1[:,2], s=0.5, c=dd[sub_idx1], alpha = 1)
plot_vector(ax, ipsc[idx1], 0.3*a1, color = 'red')
plot_vector(ax, ipsc[idx1], -0.3*b1)

ax.text(ipsc[idx1][0] + 0.5*a1[0], ipsc[idx1][1] + 0.5*a1[1], ipsc[idx1][2] + 0.5*a1[2], 
        "Principal Direction", color='r', fontsize=14)

ax.text(ipsc[idx1][0] - 0.4*b1[0], ipsc[idx1][1] - 0.4*b1[1], ipsc[idx1][2] - 0.4*b1[2], 
        "Principal Direction", color='r', fontsize=14)

# Add colorbar with shrink parameter
#cbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=-0.08)  # Adjust the shrink value as needed

# Add label to the colorbar
#cbar.set_label('Color Label')

# Set plot title
#ax.set_title("Curvature on IPSC data", y =0.97)

# Rotate the view
ax.view_init(-90, 0)
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_zticks([])
plt.axis('off')
plt.savefig('ipsc_curvature_1.png')
plt.show()