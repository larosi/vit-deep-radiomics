# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:25:38 2024

@author: Mico
"""

import os
import pandas as pd

if __name__ == "__main__":
    datasets = ['santa_maria_dataset', 'stanford_dataset']
    feature_dir = os.path.join('..', 'data', 'features')
    output_path = os.path.join(feature_dir, 'petct.parquet')

    df = []
    for dataset in datasets:
        dataset_features_dir = os.path.join(feature_dir, dataset)
        if os.path.exists(dataset_features_dir):
            df_fns = os.listdir(dataset_features_dir)
            for df_fn in df_fns:
                df_path = os.path.join(dataset_features_dir, df_fn)
                df_aux = pd.read_parquet(df_path)
                df.append(df_aux)
    df = pd.concat(df)
    df.reset_index(drop=True)

    df.to_parquet(output_path)
