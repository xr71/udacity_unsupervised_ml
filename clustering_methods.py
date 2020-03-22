import pandas as pd
import numpy as np
from sklearn import cluster
from helpers2 import simulate_data
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


dataset = simulate_data()


if __name__ == "__main__":
    print(cluster)
    print(GaussianMixture)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].scatter(x=dataset[:,0], y=dataset[:,1])
    gmm = GaussianMixture(n_components=3)

    gmm.fit(dataset)

    print(gmm.predict(dataset))

    cmap = {0:"red", 1:"green", 2:"blue"}

    ax[1].scatter(x=dataset[:,0], y=dataset[:,1],
                  c=[cmap[v] for v in gmm.predict(dataset)])

    plt.show()
