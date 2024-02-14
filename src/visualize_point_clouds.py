# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:20:20 2024

@author: Mico
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def export_to_cloud_compare(df, patients, offset=100):
    nrows = int(np.sqrt(len(patients))) + 1

    for i, row in tqdm(patients.iterrows(), total=patients.shape[0]):
        patient_id = row['patient_id']
        label = row['label']
        df_sample = df[df['patient_id'] == patient_id][['x', 'y', 'z', 'grey', 'modality', 'label']]
        df_sample['x'] = df_sample['x'] + (i // nrows) * offset + label*offset
        df_sample['y'] = df_sample['y'] + (i % nrows) * offset

        for modality in df_sample['modality'].unique():
            df_sample_path = os.path.join('..', 'data', 'points', dataset, modality, f'{patient_id}_{label}.txt')
            df_sample[df_sample['modality'] == modality][['x', 'y', 'z', 'grey']].to_csv(df_sample_path, sep=' ', index=False)

def export_umap_to_cloud_compare(df, df_umap, dataset, modality='ct', offset=10, use_2D=False, to_sketchfab=False):
    if use_2D:
        pca = PCA(n_components=2)

        df_umap = df_umap.groupby('patient_id').mean()
        df_umap[['x', 'y']] = pca.fit_transform(df_umap.values)
        df_umap['z'] = 0
    else:
        df_umap[['x', 'y', 'z']] = df_umap[['umap_x', 'umap_y', 'umap_z']]

    # scale distances to avoid overlap between patients
    distances = pairwise_distances(df_umap.values)
    min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_distance = distances[min_distance_idx]
    scale_factor = offset / min_distance
    df_umap *= scale_factor
    df_umap.sort_index(inplace=True)

    df = df[df['modality'] == modality]

    df = df.set_index('patient_id')
    df.sort_index(inplace=True)

    for coord in ['x', 'y', 'z']:
        df[coord] = df[coord] + df_umap[coord]

    if to_sketchfab:
        umap_cc_path = os.path.join('..', 'data', 'points', f'{dataset}_umap.asc')
        df[['x', 'y', 'z', 'grey', 'grey', 'grey']].astype(int).to_csv(umap_cc_path, sep=' ', index=False, header=False)
    else:
        umap_cc_path = os.path.join('..', 'data', 'points', f'{dataset}_umap.txt')
        df[['x', 'y', 'z', 'grey', 'label']].to_csv(umap_cc_path, sep=' ', index=False)

def pairwise_distances(points):
    num_points = points.shape[0]
    expanded_points = np.expand_dims(points, axis=1)
    distances = np.sqrt(np.sum((expanded_points - points)**2, axis=2))
    distances[np.arange(num_points), np.arange(num_points)] = np.inf
    return distances


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join('..', 'data', 'petct_pointcloud.parquet'))

    dataset = "santa_maria"
    #dataset = "stanford"
    df = df[df["dataset"] == dataset]

    df = df[df['mask'] > 0]
    df['grey'] = (df['norm']*255).astype(int)

    patients = df[['patient_id', 'label']].drop_duplicates()
    patients = patients.sort_values(by='label')
    patients.reset_index(drop=True, inplace=True)

    export_to_cloud_compare(df, patients, offset=100)

    df_umap = pd.read_parquet(os.path.join('..', 'data', 'petct_embeddings_umap.parquet'))
    df_umap = df_umap[['patient_id', 'umap_x', 'umap_y', 'umap_z']]
    df_umap = df_umap[df_umap['patient_id'].isin(patients['patient_id'].values)]

    export_umap_to_cloud_compare(df, df_umap, dataset, modality='ct', offset=10, use_2D=True, to_sketchfab=False)
