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
                              get_y_true_and_pred,
                              find_divisor)
from umap.umap_ import UMAP
import plotly.express as px

backbone = 'medsam'
#arch = 'transformer'
#arg_dataset = 'stanford'
#modality = 'pet'

archs = ['conv', 'transformer']
datasets = ['santa_maria', 'stanford']
modalities = ['pet', 'ct']

all_experiments = []
for arch in archs:
    for arg_dataset in datasets:
        for modality in modalities:
            all_experiments.append((arch, arg_dataset, modality))

# all_experiments = ['transformer', 'stanford', 'ct']

for arch, arg_dataset, modality in all_experiments:
    df_metrics_path = os.path.join('..', 'metrics', 'petct_metrics_sumary.csv')
    hdf5_path = os.path.join('..', 'data', 'features', f'features_masks_{modality}.hdf5')
    df_path = os.path.join('..', 'data', 'features', 'petct.parquet')
    models_save_dir = os.path.join('..', 'models', 'petct', f'{backbone}_{arch}_{arg_dataset}')
    
    desired_datasets = [arg_dataset]
    df = pd.read_parquet(df_path)  # TODO: create a function to preprocess the dataframe
    df = df[df['dataset'].isin(desired_datasets)]
    df = df[df['modality'] == modality]
    df['flip'] = df['flip'].astype(str)
    
    df['divisor'] = 1
    slices_per_patient = df.groupby(['patient_id'])['slice', 'divisor'].max()
    slices_per_patient.describe()
    slices_per_patient['divisor'] = slices_per_patient['slice'].apply(find_divisor)
    slices_per_patient = slices_per_patient['divisor'].to_dict()
    df['divisor'] = df['patient_id'].apply(lambda x: slices_per_patient[x])

    df['patient_id_new'] = df.apply(lambda row: f"{row['patient_id']}:{np.ceil(row['slice']/row['divisor']).astype(int)}", axis=1)

    df.reset_index(drop=True, inplace=True)

    # chose the model weights from the kfold and epoch
    df_metrics = pd.read_csv(df_metrics_path)
    df_metrics = df_metrics[df_metrics['Dataset'].isin([d.replace('_',' ') for d in desired_datasets])]
    df_metrics = df_metrics[df_metrics['Modality']==modality]
    df_metrics = df_metrics[df_metrics['Model'].str.contains(arch)]
    print(df_metrics.T)
    [[kfold, epoch]] = df_metrics.head(n=1)[['Best Kfold', 'Best Epoch']].values 

    # create the label encoder
    EGFR_names = list(df['label'].unique())
    EGFR_names.sort()
    EGFR_lm, EGFR_lm_inv = create_labelmap(EGFR_names)

    EGFR_encoder = OneHotEncoder(handle_unknown='ignore')
    EGFR_encoder.fit(np.array(list(EGFR_lm.keys())).reshape(-1, 1))

    #  TODO: read model parameters from a conf file
    num_layers = 1
    batch_size = 1
    feature_dim = 256
    div = 2
    if modality == 'ct':
        num_heads = 8
        dim_feedforward = feature_dim*4
    else:
        num_heads = 16
        dim_feedforward = feature_dim*2

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

    embeddings = np.stack(embeddings, axis=0)

    umap = UMAP(n_neighbors=3, min_dist=0.5, n_components=3, random_state=42, metric='l2', n_epochs=500)
    umap_features = umap.fit_transform(embeddings)

    df_umap = pd.DataFrame()
    df_umap['y_true'] = y_true
    df_umap['y_pred'] = y_pred
    df_umap['y_true'] = df_umap['y_true'].astype(str)
    df_umap['y_pred'] = df_umap['y_pred'].astype(str)

    df_umap[['umap_x', 'umap_y', 'umap_z']] = umap_features
    df_umap['id'] = df['patient_id_new'].unique()
    df_umap['id'] = df_umap['id'].str.split(':').str[0]

    fig = px.scatter_3d(df_umap, x='umap_x', y='umap_y', z='umap_z', color='y_true', symbol='id',
                        title=f'{backbone} {arch}<br>{arg_dataset} {modality}')
    fig.write_html(os.path.join('..', 'plots', 'umap', f'{backbone}_{arch}_{arg_dataset}_{modality}_umap.html'))
