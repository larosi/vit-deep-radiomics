# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:15:24 2024

@author: Mico
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from train_petct_conv import (NoduleClassifier,
                              TransformerNoduleClassifier,
                              PETCTDataset3D,
                              create_labelmap,
                              get_y_true_and_pred)
from umap.umap_ import UMAP
import plotly.express as px

backbone = 'medsam'
arch = 'conv'
arch = 'transformer'
arg_dataset = 'stanford'
modality = 'ct'

hdf5_path = os.path.join('..', 'data', 'features', f'features_masks_{modality}.hdf5')
df_path = os.path.join('..', 'data', 'features', 'petct.parquet')
models_save_dir = os.path.join('..', 'models', 'petct', f'{backbone}_{arch}_{arg_dataset}')


desired_datasets = [arg_dataset]
df = pd.read_parquet(df_path)
df = df[df['dataset'].isin(desired_datasets)]
df = df[df['modality'] == modality]
df['flip'] = df['flip'].astype(str)
df.reset_index(drop=True, inplace=True)


EGFR_names = list(df['label'].unique())
EGFR_names.sort()
EGFR_lm, EGFR_lm_inv = create_labelmap(EGFR_names)

EGFR_encoder = OneHotEncoder(handle_unknown='ignore')
EGFR_encoder.fit(np.array(list(EGFR_lm.keys())).reshape(-1, 1))
    


if modality == 'ct':
    batch_size = 1  # TODO: add support for bigger batches using zero padding to create batches of the same size
    start_epoch = 0
    num_epochs = 25
    learning_rate = 0.0001
    num_layers = 1

    if backbone == 'medsam':
        feature_dim = 256
        div = 2  # reduction factor of the conv layers

        # FIXME: deprecated transformer params
        num_heads = 8
        dim_feedforward = feature_dim*4

    else:  # dinov2
        feature_dim = 384
        div = 3  # reduction factor of the conv layers
        # FIXME: deprecated transformer params
        num_heads = 12
        dim_feedforward = feature_dim*2

else:  # TODO: PET create CNN arch for PET
    batch_size = 32
    start_epoch = 0
    num_epochs = 35
    feature_dim = 256
    learning_rate = 0.001
    num_layers = 3
    num_heads = 4
    dim_feedforward = feature_dim*4
# Create model instance
device = f'cuda:{torch.cuda.current_device()}'
if arch == 'transformer':
    model = TransformerNoduleClassifier(input_dim=feature_dim,
                                        dim_feedforward=dim_feedforward,
                                        num_heads=num_heads,
                                        num_classes=len(EGFR_lm),
                                        num_layers=num_layers)
else:
    model = NoduleClassifier(input_dim=feature_dim, num_classes=len(EGFR_lm), div=div)

kfold = '3'
epoch = 18

kfold = '0'
epoch = 8

model.load_checkpoint(os.path.join(models_save_dir, modality, f'kfold_{kfold}'), epoch)
model = model.to(device)
print(model)

dataset = PETCTDataset3D(df, 
                         label_encoder=EGFR_encoder,
                         hdf5_path=hdf5_path,
                         use_augmentation=False,
                         feature_dim=feature_dim,
                         arch=arch)

datset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

y_true = []
y_score = []
embeddings = []
model.eval()
with torch.no_grad():
    for features_batch, labels_batch in tqdm(datset_loader, desc='predict batch'):
        features_batch = features_batch.to(device)
        labels_batch = torch.squeeze(labels_batch).to(device)
        outputs = model(features_batch)

        y_true_batch, y_score_batch = get_y_true_and_pred(labels_batch, outputs[0], cpu=True)

        y_true.append(y_true_batch)
        y_score.append(y_score_batch)
        embeddings.append(torch.squeeze(outputs[1]).detach().cpu().numpy())

y_true = np.concatenate(y_true)
y_score = np.concatenate(y_score)
y_pred = (y_score[:, 1] > 0.5)*1
if arch=='transformer':
    embeddings = [f[0,:] for f in embeddings]
embeddings = np.stack(embeddings, axis=0)

umap = UMAP(n_neighbors=8, min_dist=0.0, n_components=3, random_state=42, metric='l2', n_epochs=500)
umap_features = umap.fit_transform(embeddings)

df_umap = pd.DataFrame()
df_umap['y_true'] = y_true
df_umap['y_pred'] = y_pred
df_umap['y_true'] = df_umap['y_true'].astype(str)
df_umap['y_pred'] = df_umap['y_pred'].astype(str)

df_umap[['umap_x', 'umap_y', 'umap_z']] = umap_features

fig = px.scatter_3d(df_umap, x='umap_x', y='umap_y', z='umap_z', color='y_true')
fig.write_html(os.path.join('..', 'plots', f'{backbone}_{arch}_{arg_dataset}_umap.html'))



