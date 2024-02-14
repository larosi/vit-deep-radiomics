# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:48:44 2024

@author: Mico
"""

import os
import pandas as pd
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from tfds_dense_descriptor import apply_window_ct, tfds2voxels

def to_pointcloud_df(img, mask, label, spatial_res):
    x, y, z = np.meshgrid(np.arange(0, img.shape[0]),
                          np.arange(0, img.shape[1]),
                          np.arange(0, img.shape[2]))
    df = pd.DataFrame()
    df['x'] = x.flatten() * spatial_res[0]
    df['y'] = y.flatten() * spatial_res[1]
    df['z'] = z.flatten() * spatial_res[2]
    df['raw'] = img.flatten()
    df['mask'] = mask.flatten()

    mask_box = df[df['mask'] > 0][['x', 'y', 'z']].agg(['min', 'max'])
    cond_x = (df['x'] >= mask_box.loc['min', 'x']) & (df['x'] <= mask_box.loc['max', 'x'])
    cond_y = (df['y'] >= mask_box.loc['min', 'y']) & (df['y'] <= mask_box.loc['max', 'y'])
    cond_z = (df['z'] >= mask_box.loc['min', 'z']) & (df['z'] <= mask_box.loc['max', 'z'])
    df['mask_box'] = cond_x & cond_y & cond_z
    return df


if __name__ == "__main__":
    dfs = []
    dataset_path = os.path.join('..', 'data', 'lung_radiomics')
    datasets = ['stanford_dataset', 'santa_maria_dataset']

    for dataset_name in datasets:
        if dataset_name == 'stanford_dataset':
            ds_pet, info_pet = tfds.load(f'{dataset_name}/pet', data_dir=dataset_path, with_info=True)
            ds_ct, info_ct = tfds.load(f'{dataset_name}/ct', data_dir=dataset_path, with_info=True)
        else:
            ds_pet, info_pet = tfds.load(f'{dataset_name}/pet', data_dir=dataset_path, with_info=True)
            ds_ct, info_ct = tfds.load(f'{dataset_name}/torax3d', data_dir=dataset_path, with_info=True)

        patient_pet = set(list(ds_pet.keys()))
        patient_ct = set(list(ds_ct.keys()))

        patient_ids = list(patient_ct.intersection(patient_pet))

        for patient_id in tqdm(patient_ids, desc=dataset_name):
            for modality in ['pet', 'ct']:
                if modality == 'ct':
                    img_raw, mask_raw, label, spatial_res = tfds2voxels(ds_ct, patient_id)
                else:
                    img_raw, mask_raw, label, spatial_res = tfds2voxels(ds_pet, patient_id, pet=True)

                label = label[0]
                if label not in [0, 1]:  # ignore unknown (2) and not collected (3) labels
                    print(f'nWarning: skip {patient_id} with label {label}')
                else:
                    nodule_pixels = mask_raw.sum(axis=(0, 1)).round(2)
                    if not nodule_pixels.max():
                        print(f'\nWarning: {patient_id} has empty mask')

                    df = to_pointcloud_df(img_raw, mask_raw, label, spatial_res)
                    df['modality'] = modality
                    # normalize pixel values
                    if modality == 'ct':
                        img_raw = apply_window_ct(img_raw, width=800, level=40)
                    else:
                        img_raw = img_raw / img_raw.max()
                    df['norm'] = img_raw.flatten()

                    df['dataset'] = dataset_name.replace('_dataset', '')
                    df['patient_id'] = patient_id
                    df = df[df['mask_box']]
                    df['label'] = label
                    df.reset_index(drop=True, inplace=True)
                    df[['x', 'y', 'z']] = df[['x', 'y', 'z']] - df[['x', 'y', 'z']].mean(axis=0)
                    dfs.append(df)

    dfs = pd.concat(dfs)
    dfs.to_parquet(os.path.join('..', 'data', 'petct_pointcloud.parquet'))
