# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:33:35 2024

@author: Mico
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import (accuracy_score, 
                             balanced_accuracy_score,
                             roc_auc_score,
                             f1_score)

from sklearn.neural_network import MLPClassifier

df = pd.read_parquet(os.path.join('..', 'data', 'petct_embeddings_umap.parquet'))
df['y_true'] = df['y_true'].astype(int)
df['y_pred'] = df['y_pred'].astype(int)
df_plot = df[['dataset', 'modality', 'arch', 'y_true', 'y_score']]
df_plot = df_plot.set_index(['dataset', 'modality', 'arch'])

list_fpr = []
list_tpr = []
list_models = []
list_f1 = []
list_th = []

for multiindex in df_plot.index.unique():
    y_true = df_plot.loc[multiindex][['y_true']].astype(int).values
    y_score = df_plot.loc[multiindex][['y_score']].values
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    for th in thresholds:
        f1_th = f1_score(y_true, y_score>th)
        list_f1.append(f1_th)
    list_th += thresholds.tolist() 
    list_fpr += fpr.tolist()
    list_tpr += tpr.tolist()
    auc = np.round(roc_auc_score(y_true, y_score), 3)
    list_models += [f'{multiindex[0]} {multiindex[1]}<br>{multiindex[2]} AUC {auc}']*len(tpr)

df_roc = pd.DataFrame()
df_roc['model'] = list_models
df_roc['F1 Score'] = list_f1
df_roc['Threshold'] = list_th
df_roc['False Positive Rate'] = list_fpr
df_roc['True Positive Rate'] = list_tpr

fig = px.area(df_roc, x='False Positive Rate',
              y='True Positive Rate',
              hover_data=['Threshold', 'F1 Score'],
              animation_frame='model')
fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.write_html('ROC Curve.html')


data = df[np.logical_and(df['dataset'] == 'santa_maria', df['arch'] == 'transformer')]
#data = df[np.logical_and(df['dataset'] == 'stanford', df['arch'] == 'conv')]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
patients_labels = data.groupby(['patient_id'])['y_true'].first()
patients = patients_labels.index.to_list()
patients_labels = patients_labels.to_list()

multimodal_embeddings = {'patient_id': [],
                         'pet': [],
                         'ct': []}

for patient_id in patients:
    mask_pet = np.logical_and((data['patient_id'] == patient_id), data['modality'] == 'pet')
    mask_ct = np.logical_and((data['patient_id'] == patient_id), data['modality'] == 'ct')
    pet_embed = np.stack(data.loc[mask_pet,'embeddings'].to_list()).mean(axis=0)
    ct_embed = np.stack(data.loc[mask_ct,'embeddings'].to_list()).mean(axis=0)

    multimodal_embeddings['patient_id'].append(patient_id)
    multimodal_embeddings['pet'].append(pet_embed)
    multimodal_embeddings['ct'].append(ct_embed)

df_petct = pd.DataFrame(multimodal_embeddings)
df_petct['y_true'] = patients_labels
df_petct['y_pred'] = data.groupby(['patient_id'])['y_pred'].mean().values
df_petct['y_pred']  = (df_petct['y_pred'] > 0.5)*1
        
for kfold, (train_indices, test_indices) in tqdm(enumerate(skf.split(patients, patients_labels)), desc='kfold', leave=False, position=0):   
    # get patient_ids of each split
    training_patients = [patients[i] for i in train_indices]
    testing_patients = [patients[i] for i in test_indices]
    
    
    df_train = df_petct[df_petct['patient_id'].isin(training_patients)]
    df_test = df_petct[df_petct['patient_id'].isin(testing_patients)]
    
    X_train = np.hstack([np.stack(df_train['pet'].to_list()), np.stack(df_train['ct'].to_list())])
    X_test = np.hstack([np.stack(df_test['pet'].to_list()), np.stack(df_test['ct'].to_list())])

    y_train = df_train['y_true'].astype(int).values.reshape(-1, 1)
    y_test  = df_test['y_true'].astype(int).values.reshape(-1, 1)
    
    clf = MLPClassifier(hidden_layer_sizes=(512,), solver='lbfgs', 
                        activation='logistic', early_stopping=False, validation_fraction=0.3)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, output_dict=False))
    print(classification_report(y_test, df_test['y_pred'], output_dict=False))
