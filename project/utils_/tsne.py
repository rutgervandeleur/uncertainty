import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def tsne(priors, data_point, mc_sample_num, latent_dim):
    vis = np.array([prior.sample([mc_sample_num]).cpu().detach().numpy() for prior in priors])
    vis = np.append(vis,data_point.cpu().detach().numpy())
    vis = vis.reshape((5*mc_sample_num, latent_dim))
    colors = ["red" for x in range(mc_sample_num)] + ["green" for x in range(mc_sample_num)] + ["blue" for x in range(mc_sample_num)] + ["yellow" for x in range(mc_sample_num)]  + ["black" for x in range(mc_sample_num)]

    X_embedded = PCA(n_components=2).fit_transform(vis)
    plt.figure()
    plt.scatter(X_embedded[ :,0], X_embedded[:, 1], c = colors)
    plt.savefig("tsne.png")



