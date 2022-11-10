import umap
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from IPython.display import Image
import kaleido


burst_names = [i for i in os.listdir('../data/raw_data') if 'fth' in i]
t90_duration = pd.read_csv('t90_duration.csv')


def plot_2d_tsne(X):
    plt.figure(figsize=(10,5))
    plt.scatter(pd.DataFrame(X)['x'], pd.DataFrame(X)['y'], c = X['duration_log'])
    plt.colorbar()
    plt.title('t-SNE mapping of Swift light curves, colored based on duration logT90')
    plt.show()
    
    
def plot_2d_plotly(X):
    fig = px.scatter(X, x="x", y="y", color='duration_log')
    fig.show()
    img_bytes = fig.to_image(format="png")
    Image(img_bytes)


def plot_3d_plotly(X):
    df = px.data.iris()
    fig = px.scatter_3d(X, x="x", y="y", z = 'z', color='duration_log')
    fig.show()
    img_bytes = fig.to_image(format="png")
    Image(img_bytes)

    
def fit_tsne(a, perplexity, n_components):
    tsne = TSNE(n_components= n_components, perplexity = perplexity, random_state = 42)
    X = tsne.fit_transform(a)
    X = pd.DataFrame(X)
    X['name'] = [ x.split('.')[0] for x in  burst_names]
    X = X.merge(t90_duration[['index', 'duration']], left_on='name', right_on='index')
    X['duration_log'] = np.log(X['duration'])
    
    if n_components == 2:
        X.columns = ['x', 'y', 'name', 'index', 'T90', 'duration_log']
    elif n_components == 3:
        X.columns = ['x', 'y', 'z', 'name', 'index', 'T90', 'duration_log'] 
    else :
        X.columns = [str(i) for i in X.columns]
    return X


def fit_umap(a, n_neighbors, n_components):
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
    X = reducer.fit_transform(a)
    X = pd.DataFrame(X)
    X['name'] = [ x.split('.')[0] for x in  burst_names]
    X = X.merge(t90_duration[['index', 'duration']], left_on='name', right_on='index')
    X['duration_log'] = np.log(X['duration'])
    
    if n_components == 2:
        X.columns = ['x', 'y', 'name', 'index', 'T90', 'duration_log']
    elif n_components == 3:
        X.columns = ['x', 'y', 'z', 'name', 'index', 'T90', 'duration_log'] 
    else :
        X.columns = [str(i) for i in X.columns]
    return X


def plot_clusters(X):
    plt.figure(figsize=(10,5))
    plt.scatter(pd.DataFrame(X)['x'], pd.DataFrame(X)['y'], c = X['cluster'])
    plt.colorbar()
    plt.title('t-SNE mapping of Swift light curves, colored based on clustering')
    plt.show()
