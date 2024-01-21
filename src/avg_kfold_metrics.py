# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:49:59 2024

@author: Mico
"""

import os
import pandas as pd
import json
import numpy as np


def load_json(json_path: str):
    """ load a json file from path """
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data


modality = 'ct'
kfold = 4

json_metrics = []
for k in range(0, kfold+1):
    kfold_dir = os.path.join('..', 'models', 'petct', modality, f'kfold_{k}')
    json_fns = [fn for fn in os.listdir(kfold_dir) if '.json' in fn]
    for fn in json_fns:
        json_path = os.path.join(kfold_dir, fn)
        data = load_json(json_path)
        epoch = int(fn.split('.json')[0].split('_')[-1])
        data['epoch'] = epoch
        if 'test' in fn:
            data['split'] = 'test'
        else:
            data['split'] = 'train'
        json_metrics.append(data)

df_metrics = [pd.DataFrame(split_metrics).reset_index(drop=False) for split_metrics in json_metrics]
df_metrics = pd.concat(df_metrics, axis=0)

df_metrics_group = df_metrics.set_index(['kfold', 'epoch'])
df_metrics['gmean'] = np.sqrt(df_metrics['0'] * df_metrics['1'])


df_best = df_metrics.copy()

df_best = df_best[df_best['index'] == 'recall']

df_test = df_best[df_best['split'] == 'test']
df_test = df_test.sort_values('gmean', ascending=False)
df_best_kfolds = df_test.groupby('kfold').first()

model_avg_metrics = df_best_kfolds.mean(axis=0)


best_metrics = []
for best_k, row in df_best_kfolds.iterrows():
    best_epoch = row['epoch']
    best_metrics.append(df_metrics_group.loc[(best_k, best_epoch)])

best_metrics = pd.concat(best_metrics, axis=0)
best_metrics.round(3).to_csv('best_metrics.csv')

model_avg_metrics = best_metrics.groupby(['split', 'index']).mean()
model_avg_metrics.T.round(3).to_csv('model_avg_metrics.csv')
