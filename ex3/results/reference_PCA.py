import numpy as np
import matplotlib
# LOAD THIS IF YOU RUN ON A CLUSTER
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


# CODE TO RUN REFERENCE PCA SOLUTION
if __name__ == "__main__":
  method = "PCA_PYTHON" # REFERENCE SOLUTION (COMPARE YOUR CODE WITH THIS)

  for dataset in ["2D", "faces"]:
  # for dataset in ["faces"]:
  # for dataset in ["2D"]:

    # dataset = "2D"
    # dataset = "faces"

    if dataset == "2D":
      DIM = 2
      n_components=2
    elif dataset == "faces":
      DIM = 1850
      H = 50
      W = 37
      n_components=10

    data = np.loadtxt("../data/{:}_dataset.txt".format(dataset), delimiter=',')

    N, D = np.shape(data)

    data_mean = np.mean(data,axis=0)
    data_std = np.std(data,axis=0)
    data_centered = data - data_mean
    if dataset == "faces":
      # For the faces dataset the relative input scaling is not important
      data_centered = data_centered/data_std


    # print(np.shape(data_centered))
    cov = np.matmul(data_centered.T, data_centered) / (N-1)
    # print(cov)
    eig, v = np.linalg.eig(cov)

    eig = np.abs(np.real(eig))
    print("True eigenvalues:")
    print(eig)
    np.savetxt("./data/{:}_eigs_true.txt".format(dataset), eig)

    pca_python = PCA(n_components=n_components, svd_solver='auto').fit(data_centered)
    pca_components = pca_python.components_
    eig_pred = pca_python.explained_variance_

    np.savetxt("./data/{:}{:}_components.txt".format(dataset, method), pca_components, delimiter=',')
    np.savetxt("./data/{:}{:}_mean.txt".format(dataset, method), data_mean, delimiter=',')
    np.savetxt("./data/{:}{:}_std.txt".format(dataset, method), data_std, delimiter=',')
    np.savetxt("./data/{:}{:}_eig.txt".format(dataset, method), eig_pred, delimiter=',')



    if dataset == "faces":
      # Reconstrct the images in the faces dataset
      data_reduced = pca_python.transform(data_centered)
      data_centered_rec = pca_python.inverse_transform(data_reduced)
      data_rec =  data_centered_rec*data_std + data_mean
      np.savetxt("./data/{:}{:}_data_reconstructed.txt".format(dataset, method), data_rec, delimiter=',')



