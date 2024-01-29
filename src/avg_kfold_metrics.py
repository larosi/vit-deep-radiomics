# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:49:59 2024

@author: Mico
"""

import os
import pandas as pd
import json
import plotly.express as px
import warnings
warnings.simplefilter(action='ignore')


def load_json(json_path: str):
    """ Load a json file from path """
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data


if __name__ == "__main__":
    kfold = 4
    experiments = ['medsam_transformer_stanford',
                   'medsam_transformer_santa_maria',
                   'medsam_conv_santa_maria',
                   'medsam_conv_stanford']
    modalites = ['pet', 'ct']
    folder = 'petct'

    metrics_sumary = {}
    metrics_sumary['Dataset'] = []
    metrics_sumary['Split'] = []
    metrics_sumary['Model'] = []
    metrics_sumary['Modality'] = []

    metrics_sumary['Accuracy'] = []
    metrics_sumary['AUC'] = []
    metrics_sumary['Precision'] = []
    metrics_sumary['Recall'] = []

    metrics_sumary['Best Kfold'] = []
    metrics_sumary['Best Epoch'] = []

    for modality in modalites:
        for experiment in experiments:
            json_metrics = []
            for k in range(0, kfold+1):
                kfold_dir = os.path.join('..', 'models', folder, experiment, modality, f'kfold_{k}')
                if os.path.exists(kfold_dir):
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

            # save the training loss and accuracy curves with an animation slider to select the Kfold
            df_metrics = [pd.DataFrame(split_metrics).reset_index(drop=False) for split_metrics in json_metrics]
            df_metrics = pd.concat(df_metrics, axis=0)
            df_metrics_group = df_metrics.set_index(['kfold', 'epoch'])

            fig = px.line(df_metrics[df_metrics['index'] == 'recall'].sort_values('epoch'),
                          x='epoch', y='loss', color='split', animation_frame='kfold', title=experiment, markers=True)
            fig.update_yaxes(range=[0, df_metrics['loss'].max()+0.1])
            fig.update_xaxes(range=[0, df_metrics['epoch'].max()+1])
            fig.write_html(os.path.join('..', 'plots', f'{experiment}-{modality}-training_loss.html'))

            fig = px.line(df_metrics[df_metrics['index'] == 'recall'].sort_values('epoch'),
                          x='epoch', y='accuracy', color='split', animation_frame='kfold', title=experiment, markers=True)
            fig.update_yaxes(range=[0, df_metrics['accuracy'].max()+0.1])
            fig.update_xaxes(range=[0, df_metrics['epoch'].max()+1])
            fig.write_html(os.path.join('..', 'plots', f'{experiment}-{modality}-training_accuracy.html'))

            # select the best model of each kfold based on a target metric
            df_best = df_metrics.copy()
            df_best = df_best[df_best['index'] == 'f1-score']
            df_best = df_best[df_best['split'] == 'test']
            df_best['target_metric'] = df_best['ROC AUC']
            df_best = df_best.sort_values('target_metric', ascending=False)
            df_best_kfolds = df_best.groupby('kfold').first()

            model_avg_metrics = df_best_kfolds.mean(axis=0)

            best_metrics = []
            for best_k, row in df_best_kfolds.iterrows():
                best_epoch = row['epoch']
                best_metrics.append(df_metrics_group.loc[(best_k, best_epoch)])

            best_metrics = pd.concat(best_metrics, axis=0)
            #best_metrics.round(3).to_csv('best_metrics.csv')

            # mean across all kfolds of the best models
            model_avg_metrics = best_metrics.groupby(['split', 'index']).mean()
            model_std_metrics = best_metrics.groupby(['split', 'index']).std()   # TODO: show the std of each metric
            #model_avg_metrics.T.round(3).to_csv('model_avg_metrics.csv')

            best_k, best_epoch = best_metrics.iloc[best_metrics['ROC AUC'].argmax()].name

            # Store the kfold avg metrics of each 'Dataset', 'Model', 'Modality', 'Split'
            for split in ['train', 'test']:
                auc = model_avg_metrics.loc[split, 'recall']['ROC AUC']
                accuracy = model_avg_metrics.loc[split, 'recall']['accuracy']
                precision = model_avg_metrics.loc[split, 'precision']['1']
                recall = model_avg_metrics.loc[split, 'recall']['1']

                model_name = ' '.join(experiment.split('_')[0:2])
                dataset = ' '.join(experiment.split('_')[2:])

                metrics_sumary['Dataset'].append(dataset)
                metrics_sumary['Split'].append(split)
                metrics_sumary['Model'].append(model_name)
                metrics_sumary['Modality'].append(modality)

                metrics_sumary['Accuracy'].append(accuracy)
                metrics_sumary['AUC'].append(auc)
                metrics_sumary['Precision'].append(precision)
                metrics_sumary['Recall'].append(recall)

                metrics_sumary['Best Kfold'].append(best_k)
                metrics_sumary['Best Epoch'].append(best_epoch)

            print(f'\n{dataset} {model_name} {modality}')
            print(model_avg_metrics[['0', '1', 'accuracy', 'ROC AUC', 'loss']].round(2))

    df_metrics_sumary = pd.DataFrame(metrics_sumary)
    sumary_index = ['Dataset', 'Model', 'Modality', 'Split']
    df_metrics_sumary = df_metrics_sumary.set_index(sumary_index).sort_index()
    df_metrics_sumary = df_metrics_sumary.sort_index(level=[0, 1, 2, 3], ascending=[True, True, True, False])
    df_metrics_sumary.round(3).to_csv(os.path.join('..', 'metrics', f'{folder}_metrics_sumary.csv'))
