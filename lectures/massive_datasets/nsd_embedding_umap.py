# file: nsd_embedding_umap.py
import numpy as np, matplotlib.pyplot as plt, umap, h5py
with h5py.File('nsd_clip_embeddings.h5','r') as f:
    emb = f['embed'][:]           # pre-computed ViT-L/14 features
    cat = f['category'][:]        # 73 k categorical labels
reducer = umap.UMAP(n_neighbors=100, min_dist=0.8, metric='cosine')
um = reducer.fit_transform(emb[::10])  # subsample for speed
plt.scatter(um[:,0], um[:,1], c=cat[::10], s=1, cmap='tab20'); plt.colorbar(); plt.show()