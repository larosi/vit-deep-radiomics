# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:49:31 2024

@author: Mico
"""

import os
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from config_manager import get_project_dir

project_dir = get_project_dir()
df_path = os.path.join(project_dir, 'data', 'features', 'petct.parquet')
drop_patients = ['AMC-022', 'R01-064']
df = pd.read_parquet(df_path)
df = df[df['modality'] == 'ct']
df = df[~df['patient_id'].isin(drop_patients)]
df.reset_index(inplace=True, drop=True)
kfold_patients = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for dataset in df['dataset'].unique():
    patients_labels = df[df['dataset'] == dataset].groupby(['patient_id'])['label'].first()
    patients = patients_labels.index.to_list()
    patients_labels = patients_labels.to_list()
    kfold_patients[dataset] = {}
    for kfold, (train_indices, test_indices) in enumerate(skf.split(patients, patients_labels)):
        kfold_patients[dataset][kfold] = {'train': [], 'test': []}
        training_patients = [patients[i] for i in train_indices]
        testing_patients = [patients[i] for i in test_indices]

        kfold_patients[dataset][kfold]['train'] = training_patients
        kfold_patients[dataset][kfold]['test'] = testing_patients

kfold_path = os.path.join(project_dir, 'conf', 'parameters_kfold.yaml')

with open(kfold_path, "w") as f:
    data = {'kfold_patients': kfold_patients}
    yaml.dump(data, f)
