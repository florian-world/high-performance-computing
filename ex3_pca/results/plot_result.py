import numpy as np
import matplotlib
import matplotlib.gridspec
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


if __name__ == "__main__":

  # SELECT YOUR METHOD:
  # method = "PCA_PYTHON" # REFERENCE SOLUTION (COMPARE YOUR CODE WITH THIS)
  method = "PCA" # YOUR CPP IMPLEMENTATION
  # method = "OJA" # YOUR CPP IMPLEMENTATION

  # SELECT THE DATASET
  # dataset = "2D"
  dataset = "faces"

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
  data_mean_true = np.mean(data,axis=0)
  data_std_true = np.std(data,axis=0)
  data_centered = data - data_mean_true
  if dataset == "faces":
    # For the faces dataset the scaling is not important
    data_centered = data_centered/data_std_true


  eig_true = np.loadtxt("./data/{:}_eigs_true.txt".format(dataset))
  print("TRUE EIGENVALUES:")
  print(eig_true)

  # LOADING THE RESULTS
  result_str = "./data/{:}{:}_components.txt".format(dataset, method)
  pca_components = np.loadtxt(result_str, delimiter=',')
  temp = list(np.shape(pca_components))
  size = 1
  for s in temp:
    size = size*s
  size = int(size)
  n_components = int(size/D)
  pca_components = np.reshape(pca_components, (n_components, -1))
  result_str = "./data/{:}{:}_mean.txt".format(dataset, method)
  data_mean = np.loadtxt(result_str, delimiter=',')
  result_str = "./data/{:}{:}_std.txt".format(dataset, method)
  eig_pred = np.loadtxt("./data/{:}{:}_eig.txt".format(dataset, method), delimiter=',')
  data_std = np.loadtxt("./data/{:}{:}_std.txt".format(dataset, method), delimiter=',')

  if dataset == "faces":
    data_rec = np.loadtxt("./data/{:}{:}_data_reconstructed.txt".format(dataset, method), delimiter=',')
    print("SHAPES:")
    print(np.shape(data))
    print(np.shape(data_rec))



  print("PREDICTED EIGENVALUES:")
  print(eig_pred)
  if dataset == "2D":
    print("PREDICTED COMPONENTS:")
    print(pca_components)
    # print("PREDICTED MEAN:")
    # print(data_mean)
    # print("PREDICTED STD:")
    # print(data_std)



  if dataset == "2D":

    for pca_comp in pca_components:
      # For illustration purposes, we additionally scaled them with a single float (the standard deviation of the whole dataset)
      pca_comp = pca_comp * data_centered.std()
      plt.arrow(0, 0, pca_comp[0], pca_comp[1], color="red", linewidth=3, head_width=0.15, head_length=0.2, alpha=0.75, zorder=1)

    plt.plot(data_centered[:,0], data_centered[:,1], "x", color="forestgreen", zorder=0)
    # plt.title("Method: {:} Eigenmodes".format(method))
    plt.savefig("figures/{:}_{:}_components.pdf".format(dataset, method))
    plt.close()



  elif dataset == "faces":
    pca_components = pca_components.reshape((n_components, H, W))
    data_mean = data_mean.reshape((H, W))
    data_std = data_std.reshape((H, W))
    data_std_true = data_std_true.reshape((H, W))


    print(data_mean)
    print(np.shape(data_mean))
    print(np.min(data_mean))
    print(np.max(data_mean))

    n_max_plot = 8
    if n_components > n_max_plot:
      n_components = n_max_plot

    fig = plt.figure(figsize=(20,8))
    gs = fig.add_gridspec(2, n_components+1)


    ax = fig.add_subplot(gs[0, 0])
    # Plotting the mean face
    ax.imshow(data_mean, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Mean face")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(data_std, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Std face")

    ax = fig.add_subplot(gs[0, 2:])
    # Plotting the eigenvalue spectrum
    PLOTDIM=20
    ax.plot(np.arange(np.shape(eig_true[:PLOTDIM])[0]), eig_true[:PLOTDIM], "bx", label="TRUE eigenvalues")
    ax.plot(np.arange(np.shape(eig_pred[:PLOTDIM])[0]), eig_pred[:PLOTDIM], "ro", label="Computed")
    ax.legend()
    ax.set_title("Spectrum")



    for i in range(n_components):
        pca_comp = pca_components[i]
        pca_comp = pca_comp * data_std_true
        ax = fig.add_subplot(gs[1, i])

        ax.imshow(pca_comp, cmap='bone', interpolation = 'bicubic')
        ax.set_title("Eigenface {:}".format(i))

    fig.suptitle('METHOD {:}'.format(method))
    plt.savefig("figures/{:}_{:}_components.pdf".format(dataset, method))
    plt.close()

    # print(data.mean())
    # print(data.std())
    # print(data_rec.mean())
    # print(data_rec.std())

    
    # RESHAPING
    data = data.reshape((-1,H,W))
    data_rec = data_rec.reshape((-1,H,W))
    EXAMPLE_IMAGES=5
    fig, axes = plt.subplots(2, EXAMPLE_IMAGES, figsize=(18, 4))
    for i in range(EXAMPLE_IMAGES):
      axes[0,i].imshow(data[i], cmap='gray', vmin=0.0, vmax=255.0)
      axes[1,i].imshow(data_rec[i], cmap='gray', vmin=0.0, vmax=255.0)
    fig.suptitle('METHOD {:}'.format(method))
    plt.savefig("figures/{:}_{:}_reconstruction.pdf".format(dataset, method))
    plt.close()









