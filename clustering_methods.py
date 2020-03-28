import pandas as pd
import numpy as np
from sklearn import cluster
from helpers2 import simulate_data
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

dataset = simulate_data()

def run_gaussian(dataset):
    gmm = GaussianMixture(n_components=3)
    gmm.fit(dataset)
    preds = gmm.predict(dataset)

    return preds


def run_hclust(dataset):
    aggmodel = cluster.AgglomerativeClustering(n_clusters=3)
    preds = aggmodel.fit_predict(dataset)

    return preds


if __name__ == "__main__":
    print(cluster)
    print(GaussianMixture)
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].scatter(x=dataset[:,0], y=dataset[:,1])


    gmm_preds    = run_gaussian(dataset)
    hclust_preds = run_hclust(dataset)
    linkage_matrix = linkage(dataset, method="ward")

    cmap = {0:"red", 1:"green", 2:"blue"}

    ax[1].scatter(x=dataset[:,0], y=dataset[:,1],
                  c=[cmap[v] for v in gmm_preds])
    dendrogram(linkage_matrix)
    plt.show()
