# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 07:53:47 2024

@author: Mico
"""
import os
import pandas as pd
import numpy as np
import json
import h5py
from tqdm import tqdm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models_archs import save_checkpoint, TransformerNoduleBimodalClassifier
from config_manager import load_conf


def positional_encoding_3d(x, y, z, D, scale=10000):
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    n_points = x.shape[0]
    encoding = np.zeros((n_points, D))

    for i in range(D // 6):
        exponent = scale ** (6 * i / D)
        encoding[:, 2*i] = np.sin(x / exponent)
        encoding[:, 2*i + 1] = np.cos(x / exponent)
        encoding[:, 2*i + D // 3] = np.sin(y / exponent)
        encoding[:, 2*i + 1 + D // 3] = np.cos(y / exponent)
        encoding[:, 2*i + 2 * D // 3] = np.sin(z / exponent)
        encoding[:, 2*i + 1 + 2 * D // 3] = np.cos(z / exponent)

    return encoding


class PETCTDataset3D(Dataset):
    def __init__(self, dataframe, label_encoder, hdf5_ct_path, hdf5_pet_path, use_augmentation=False, feature_dim=256, arch='conv'):
        self.slice_per_modality = dataframe.groupby(['patient_id', 'modality'])['slice'].max()
        self.df_ct = dataframe[dataframe['modality'] == 'ct'].reset_index(drop=True)
        self.df_pet = dataframe[dataframe['modality'] == 'pet'].reset_index(drop=True)

        if use_augmentation:
            n_samples = len(self.df_ct['patient_id_new'].unique())
            self.dataframe = self.df_ct.copy()
            self.dataframe['patient_id_new_int'] = self.dataframe['patient_id_new'].str.split(':').str[-1]
            self.dataframe['patient_id_new_int'] = self.dataframe['patient_id_new_int'].astype(int)
            self.dataframe.sort_values(by='patient_id_new_int', inplace=True, ascending=False)
            self.dataframe = self.dataframe.groupby(['patient_id'])[['modality', 'dataset', 'label', 'patient_id_new', 'patient_id_new_int']].first()
            self.dataframe.reset_index(inplace=True, drop=False)
            repeat_times = min(max(3, int(np.ceil(n_samples / self.dataframe.shape[0]))), 1)
            self.dataframe = pd.DataFrame(np.repeat(self.dataframe.values, repeat_times, axis=0), columns=self.dataframe.columns)
        else:
            self.dataframe = self.df_ct.groupby(['patient_id_new'])[['modality', 'dataset', 'label', 'patient_id']].first()
            self.dataframe.reset_index(inplace=True, drop=False)

        self.use_augmentation = use_augmentation

        self.flip_angles = dataframe.groupby(['flip', 'angle'], as_index=False).size()[['flip', 'angle']]

        self.df_ct = self.df_ct.set_index(['patient_id_new', 'angle', 'flip'])
        self.df_pet = self.df_pet.set_index(['patient_id', 'angle', 'flip'])
        self.df_ct = self.df_ct.sort_index()
        self.df_pet = self.df_pet.sort_index()

        self.hdf5_ct_path = hdf5_ct_path
        self.hdf5_pet_path = hdf5_pet_path
        self.label_encoder = label_encoder
        self.feature_dim = feature_dim
        self.arch = arch

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        enable_random_crop = True
        noise_val = 10
        sample = self.dataframe.iloc[idx]
        patient_id_rew = sample.patient_id_new
        patient_id = sample.patient_id
        label = sample.label
        noise = np.random.random(3) * noise_val - noise_val/2
        scale_noise = np.random.uniform(0.85, 1.15)
        if self.use_augmentation:
            [[flip, angle]] = self.flip_angles.sample(n=1).values
            patient_int = sample.patient_id_new_int
            if patient_int > 0:
                patient_int = np.random.randint(0, patient_int)
            patient_id_rew = f'{patient_id}:{patient_int}'
        else:
            flip = 'None'
            angle = 0
            noise = noise * 0
            scale_noise = 1.0

        ct_slices = self.df_ct.loc[(patient_id_rew, angle, flip)]['slice'].values
        start_slice_index, end_slice_index = ct_slices.argmin(), ct_slices.argmax()
        if enable_random_crop:  # TODO: move to cfg file
            if self.use_augmentation: # random slice crop
                if len(ct_slices) > 7:
                    window_size = np.random.randint(7, len(ct_slices))
                    start_slice_index = np.random.randint(0, len(ct_slices)-window_size)
                    end_slice_index = start_slice_index + window_size

        feature_ids = self.df_ct.loc[(patient_id_rew, angle, flip)]['feature_id'].values[start_slice_index:end_slice_index]
        spatial_res = self.df_ct.loc[(patient_id_rew, angle, flip)]['spatial_res'].values[0]
        spatial_res = np.abs(spatial_res) * scale_noise
        features_ct = self._get_features(self.hdf5_ct_path, patient_id, feature_ids, angle, flip, noise, spatial_res)
        features_ct = torch.as_tensor(features_ct, dtype=torch.float32)

        ct_slices = ct_slices[start_slice_index:end_slice_index] / self.slice_per_modality.loc[(patient_id, 'ct')]
        start_slice, end_slice = ct_slices.min(), ct_slices.max()

        max_slice = self.slice_per_modality[patient_id, 'pet']
        start_slice = max(0, int(start_slice*max_slice))
        end_slice = min(max_slice, int(end_slice*max_slice))

        df_pet = self.df_pet.loc[(patient_id, angle, flip)]
        spatial_res = df_pet['spatial_res'].values[0]
        spatial_res = np.abs(spatial_res) * scale_noise
        feature_ids = df_pet[np.logical_and(df_pet['slice'] >= start_slice, df_pet['slice'] <= end_slice)]['feature_id'].values
        features_pet = self._get_features(self.hdf5_pet_path, patient_id, feature_ids, angle, flip, noise, spatial_res)
        features_pet = torch.as_tensor(features_pet, dtype=torch.float32)

        labels = np.array(label)
        labels = np.expand_dims(labels, axis=-1)
        labels = self.label_encoder.transform(labels.reshape(-1, 1)).toarray()
        labels = torch.as_tensor(labels, dtype=torch.float32)

        return features_ct, features_pet, labels

    def _get_features(self, hdf5_path, patient_id, feature_ids, angle, flip, noise, spatial_res):
        features = []
        masks = []

        with h5py.File(hdf5_path, 'r') as h5f:
            for feature_id in feature_ids:
                slice_features = h5f[f'{patient_id}/features/{feature_id}'][()]
                slice_mask_orig = h5f[f'{patient_id}/masks/{feature_id}'][()]
                slice_mask = resize(slice_mask_orig, slice_features.shape[0:2], order=0)
                slice_mask = np.expand_dims(slice_mask, axis=-1)
                if self.arch == 'conv':
                    features.append(slice_features * slice_mask)  # elementwise prod feature-mask
                else:
                    features.append(slice_features) 
                masks.append(slice_mask)

        features = np.transpose(np.stack(features, axis=0), axes=(3, 0, 1, 2))  # (slice, h, w, feat_dim) -> (feat_dim, slice, h, w)
        if self.arch == 'transformer':
            masks = np.transpose(np.stack(masks, axis=0), axes=(1, 2, 0, 3))  # (slice, h, w, 1) -> (h, w, slice, 1)
            h_orig, w_orig = slice_mask_orig.shape[0:2]
            features = np.transpose(features, axes=(2, 3, 1, 0))  # (h, w, slice, feat_dim)
            h_new, w_new = features.shape[0], features.shape[1]

            x, y, z = np.meshgrid(np.arange(0, features.shape[0]),
                                  np.arange(0, features.shape[1]),
                                  np.arange(0, features.shape[2]))
            x = (x.flatten() / w_new).flatten() * w_orig * spatial_res[0]
            y = (y.flatten() / h_new).flatten() * h_orig * spatial_res[1]
            z = (z.flatten()).flatten() * spatial_res[2]

            masks = masks.flatten()
            x = (x - x.mean() + noise[0])[masks]
            y = (y - y.mean() + noise[1])[masks]
            z = (z - z.mean() + noise[2])[masks]

            pe = positional_encoding_3d(x, y, z, D=self.feature_dim, scale=10000)

            features = features.reshape(-1, self.feature_dim)[masks, :] + pe / 4  # (seq_len, feat_dim)

        return features


def print_classification_report(report, global_metrics=None):
    """ Display a sklearn like classification_report with extra metrics

    Args:
        report (dict): Sklearn classification_report .
        global_metrics (list[str], optional): list of the extra global metrics.

    """
    df = pd.DataFrame(report)
    df = df.round(3)

    if global_metrics is None:
        global_metrics = ['accuracy', 'ROC AUC', 'kfold', 'loss', 'epoch', 'split']
    headers = df.index.to_list()

    df = df.T.astype(str)
    support = df.loc['macro avg'].iloc[-1]

    for row in global_metrics:
        metric_val = df.loc[row].iloc[-2]
        new_row = [' ']*len(headers)
        new_row[-2] = metric_val
        new_row[-1] = support
        df.loc[row] = new_row

    local_metrics = [col for col in df.T.columns if col not in global_metrics]
    df_local = df.loc[local_metrics]
    df_global = df.loc[global_metrics].T[-2:-1]
    df_global.index = ['   ']

    final_str = f'\n{df_global}\n\n{df_local}\n\n'
    print(final_str)
    return final_str


def plot_loss_metrics(df_loss, title):
    """ Plot loss vs epoch with the standard desviation between kfolds

    Args:
        df_loss (pd.dataframe): dataframe with the training loss metrics.

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure.

    """
    metric_names = ['Loss', 'AUC', 'F1', 'Target_metric']
    plot_grid = [[1, 1], [1, 2], [2, 1], [2, 2]]
    fig = make_subplots(rows=2,
                        shared_xaxes=True,
                        cols=2,
                        subplot_titles=metric_names)
    for plot_i, metric_name in enumerate(metric_names):
        metric_name = metric_name.lower()
        if f'train_{metric_name}' in df_loss.columns:
            fig.append_trace(go.Scatter(x=df_loss['epoch'],
                                        y=df_loss[f'train_{metric_name}'],
                                        mode='lines+markers',
                                        marker_color='red',
                                        name=f'train_{metric_name}',
                                        hovertext=df_loss['train_report']
                                        ),
                             row=plot_grid[plot_i][0], col=plot_grid[plot_i][1])
            fig.append_trace(go.Scatter(x=df_loss['epoch'],
                                        y=df_loss[f'test_{metric_name}'],
                                        mode='lines+markers',
                                        marker_color='blue',
                                        name=f'test_{metric_name}',
                                        hovertext=df_loss['test_report']
                                        ),
                             row=plot_grid[plot_i][0], col=plot_grid[plot_i][1])
        else:
            fig.append_trace(go.Scatter(x=df_loss['epoch'],
                                        y=df_loss[f'{metric_name}'],
                                        mode='lines+markers',
                                        marker_color='green',
                                        name=f'{metric_name}',
                                        hovertext=df_loss['is_improvement']),
                             
                             row=plot_grid[plot_i][0], col=plot_grid[plot_i][1])
    fig.update_layout(title_text=title.capitalize(), xaxis_title="Epochs",)
    return fig


def create_labelmap(label_names):
    """ Create a labelmap from a list labels

    Args:
        label_names (list): list of labels.

    Returns:
        labelmap (dict): to convert label_id to label_name.
        labelmap_inv (dict): to convert label_name to label_id.

    """
    labelmap = dict(zip(np.arange(0, len(label_names)), label_names))
    labelmap_inv = dict(zip(label_names, np.arange(0, len(label_names))))
    return labelmap, labelmap_inv


def get_y_true_and_pred(y_true, y_pred, cpu=False):
    """ Check tensor sizes and apply softmax to get y_score

    Args:
        y_true (torch.tensor): batch of one-hot encoding true labels.
        y_pred (torch.tensor): batch of prediction logits.
        cpu (bool, optional): return tensors as numpy arrays. Defaults to False.

    Returns:
        y_true (torch.tensor or np.array): true labels.
        y_score (torch.tensor or np.array): pred labels probabilities.

    """
    y_true = torch.squeeze(y_true)
    y_pred = torch.squeeze(y_pred)
    assert y_pred.size() == y_true.size()

    if len(y_true.shape) == 1:
        y_pred = torch.unsqueeze(y_pred, 0)
        y_true = torch.unsqueeze(y_true, 0)

    y_score = F.softmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)

    if cpu:
        y_true = y_true.detach().cpu().numpy()
        y_score = y_score.detach().cpu().numpy()

    return y_true, y_score


def get_sampler_weights(train_labels):
    """ Compute the sampler weights of the train dataloader

    Args:
        train_labels (np.array): labels of the train dataloader.

    Returns:
        weights (list): a weight for each element in the dataloader.

    """
    labels_mean = train_labels.mean()
    weight_for_0 = (1 - labels_mean)
    weight_for_1 = labels_mean
    weights = [1/weight_for_0 if lb == 0 else 1/weight_for_1 for lb in train_labels]
    return weights


class CrossModalFocalLoss(nn.Module):
    """
     Multi-class Cross Modal Focal Loss
    """
    def __init__(self, gamma=2, alpha=None, beta=0.5):
        super(CrossModalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8

    def forward(self, inputs_petct, inputs_ct, inputs_pet, targets):
        """
        inputs_petct: [N, C], float32
        inputs_ct: [N, C], float32
        inputs_pet: [N, C], float32
        target: [N, ], int64
        """
        if len(inputs_petct.shape) == 1:
            inputs_petct = torch.unsqueeze(inputs_petct, 0)
            inputs_ct = torch.unsqueeze(inputs_ct, 0)
            inputs_pet = torch.unsqueeze(inputs_pet, 0)
            targets = torch.unsqueeze(targets, 0)
        class_indices = torch.argmax(targets, dim=1)

        logpt_petct = F.log_softmax(inputs_petct, dim=1)
        logpt_ct = F.log_softmax(inputs_ct, dim=1)
        logpt_pet = F.log_softmax(inputs_pet, dim=1)

        pt_petct = torch.exp(logpt_petct)
        logpt_petct = (1-pt_petct)**self.gamma * logpt_petct
        loss_petct = F.nll_loss(logpt_petct, class_indices, self.alpha, reduction='mean')

        pt_ct = torch.exp(logpt_ct)
        pt_pet = torch.exp(logpt_pet)

        pt_mean = (2*pt_ct*pt_pet) / (pt_ct + pt_pet + self.eps)

        logpt_ct = (1-pt_mean*pt_ct)**self.gamma * logpt_ct
        loss_ct = F.nll_loss(logpt_ct, class_indices, self.alpha, reduction='mean')

        logpt_pet = (1-pt_mean*pt_pet)**self.gamma * logpt_pet
        loss_pet = F.nll_loss(logpt_pet, class_indices, self.alpha, reduction='mean')

        loss = (self.beta*loss_petct + (1-self.beta)*(loss_ct + loss_pet))
        return loss


class FocalLoss(nn.Module):
    """
     Multi-class Focal Loss
    """
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = alpha

    def forward(self, inputs, targets):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        if len(inputs.shape) == 1:
            inputs = torch.unsqueeze(inputs, 0)
            targets = torch.unsqueeze(targets, 0)
        class_indices = torch.argmax(targets, dim=1)

        logpt = F.log_softmax(inputs, dim=1)

        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, class_indices, self.weight, reduction='sum')
        return loss


def find_divisor(slice_count, modality):
    if modality == 'ct':
        desired_slices = 15
    else:
        desired_slices = 2
    return np.clip(desired_slices, 1, slice_count)


def prepare_df(df):
    df['divisor'] = 1
    slices_per_patient = df.groupby(['patient_id', 'modality'])[['slice', 'divisor']].max()
    slices_per_patient.describe()

    slices_per_patient['divisor'] = slices_per_patient.apply(lambda x: find_divisor(x['slice'], modality=x.name[1]), axis=1)
    slices_per_patient = slices_per_patient['divisor'].to_dict()

    df['divisor'] = df[['patient_id', 'modality']].apply(lambda x: slices_per_patient[(x[0], x[1])], axis=1) 

    df['patient_id_new'] = df.apply(lambda row: f"{row['patient_id']}:{np.ceil(row['slice']/row['divisor']).astype(int)}", axis=1)
    
    df_pet = df[df['modality'] == 'pet']
    df_ct = df[df['modality'] == 'ct']
    
    patient_ids = df_ct['patient_id'].unique()
    df_aux = []
    for patient_id in patient_ids:
        df_patient = df_ct[df_ct['patient_id'] == patient_id]
        window_size = df_patient['divisor'].max()
        slices = df_patient['slice'].unique()
        min_slice = slices.min()
        max_slice = slices.max()

        for sample_i, slice_i in enumerate(range(0, len(slices)-window_size)):
            mask = np.logical_and(df_patient['slice'] >= slice_i, df_patient['slice'] <= slice_i+window_size)
            df_patient.loc[mask,'patient_id_new'] = f'{patient_id}:{sample_i}'
            df_aux.append(df_patient[mask])
    df_ct = pd.concat(df_aux, axis=0)
    df = pd.concat([df_ct, df_pet], axis=0)
    df.reset_index(drop=True, inplace=True)

    return df

def get_number_of_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    return param_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D transoformer or CNN for lung nodules clasification")
    parser.add_argument("-a", "--arch", type=str, default="transformer",
                        help="'transformer' or 'conv'")
    parser.add_argument("-d", "--dataset", type=str, default="stanford",
                        help="dataset 'stanford' or 'santa_maria'")
    parser.add_argument("-b", "--backbone", type=str, default="medsam",
                        help="backbone ViT encoder 'medsam' or 'dinov2'")
    parser.add_argument("-m", "--modality", type=str, default="petct",
                        help="'ct', 'pet' or 'petct'")

    args = parser.parse_args()

    arch = args.arch
    backbone = args.backbone
    modality = args.modality
    use_sampler = False
    arg_dataset = args.dataset

    desired_datasets = [arg_dataset]

    if modality == 'petct':
        desired_modalities = ['pet', 'ct']
    else:
        desired_modalities = [modality]

    hdf5_pet_path = os.path.join('..', 'data', 'features', 'features_masks_pet.hdf5')
    hdf5_ct_path = os.path.join('..', 'data', 'features', 'features_masks_ct.hdf5')
    df_path = os.path.join('..', 'data', 'features', 'petct.parquet')
    #models_save_dir = os.path.join('..', 'models', 'petct')  # TODO:  generate an unique experiment ID
    models_save_dir = os.path.join('..', 'models', 'petct', f'{backbone}_{arch}_{arg_dataset}')
    
    cfg = load_conf()

    # load dataframe and apply some filter criteria
    df = pd.read_parquet(df_path)
    df= df[df['dataset'].isin(desired_datasets)]
    df = df[df['modality'].isin(desired_modalities)]
    df['flip'] = df['flip'].astype(str)
    df.reset_index(drop=True, inplace=True)
    df = prepare_df(df)

    # create labelmap and onehot enconder for nodule EGFR mutation
    EGFR_names = list(df['label'].unique())
    EGFR_names.sort()
    EGFR_lm, EGFR_lm_inv = create_labelmap(EGFR_names)

    EGFR_encoder = OneHotEncoder(handle_unknown='ignore')
    EGFR_encoder.fit(np.array(list(EGFR_lm.keys())).reshape(-1, 1))

    print('labelmap:')
    print(EGFR_lm)
    print(EGFR_encoder.transform(np.array(list(EGFR_lm.keys())).reshape(-1, 1)).toarray())

    train_metrics = {'kfold': [],
                     'epoch': [],
                     'train_loss': [],
                     'test_loss': [],
                     'train_auc': [],
                     'test_auc': [],
                     'train_f1': [],
                     'test_f1': [],
                     'train_report': [],
                     'test_report': []}

    # use KFold to split patients stratified by label
    folds = list(cfg['kfold_patients'][arg_dataset].keys())
    for kfold in tqdm(folds, desc='kfold', leave=False, position=0):   
        save_dir = os.path.join(models_save_dir, modality, f'kfold_{kfold}')
        os.makedirs(save_dir, exist_ok=True)

        # get patient_ids of each split
        training_patients = cfg['kfold_patients'][arg_dataset][kfold]['train']
        testing_patients = cfg['kfold_patients'][arg_dataset][kfold]['test']

        # filter dataframes based on the split patients
        df_train = df[df['patient_id'].isin(training_patients)]
        df_test = df[df['patient_id'].isin(testing_patients)]

        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        #  TODO: define training parameters in a .yaml file

        batch_size = 1  # TODO: add support for bigger batches using zero padding to create batches of the same size
        start_epoch = 0
        num_epochs = 50

        cfg_model = cfg['models'][arch]
        learning_rate = cfg_model['learning_rate']
        feature_dim = cfg_model['feature_dim']

        mlp_ratio_ct = cfg_model['ct']['mlp_ratio']
        mlp_ratio_pet = cfg_model['pet']['mlp_ratio']

        num_heads_ct = cfg_model['ct']['num_heads']
        num_heads_pet = cfg_model['pet']['num_heads']

        num_layers_ct = cfg_model['ct']['num_layers']
        num_layers_pet = cfg_model['pet']['num_layers']

        # Create model instance
        device = f'cuda:{torch.cuda.current_device()}'

        model = TransformerNoduleBimodalClassifier(feature_dim,
                                                   mlp_ratio_ct, mlp_ratio_pet,
                                                   num_heads_ct, num_heads_pet,
                                                   num_layers_ct, num_layers_pet,
                                                   num_classes=len(EGFR_lm))

        print(model)
        print(get_number_of_params(model))
        model = model.to(device)

        # CrossEntropyLoss because the last layer has one output per class (mutant, wildtype)

        #criterion = nn.CrossEntropyLoss()
        #criterion = FocalLoss(alpha=torch.tensor([0.25, 0.75]).to(device), gamma=2.0)
        #criterion = CrossModalFocalLoss(alpha=torch.tensor([0.25, 0.75]).to(device), gamma=2.5, beta=0.6)
        criterion = CrossModalFocalLoss(alpha=torch.tensor([0.25, 0.75]).to(device), gamma=2.5, beta=0.6)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

        # create datasets
        train_dataset = PETCTDataset3D(df_train,
                                       label_encoder=EGFR_encoder,
                                       hdf5_ct_path=hdf5_ct_path,
                                       hdf5_pet_path=hdf5_pet_path,
                                       use_augmentation=True,
                                       feature_dim=feature_dim,
                                       arch=arch)

        test_dataset = PETCTDataset3D(df_test,
                                      label_encoder=EGFR_encoder,
                                      hdf5_ct_path=hdf5_ct_path,
                                      hdf5_pet_path=hdf5_pet_path,
                                      use_augmentation=False,
                                      feature_dim=feature_dim,
                                      arch=arch)

        # create a sampler to balance training classes proportion
        if use_sampler:
            train_labels = np.array(list(train_dataset.dataframe.label.values))
            sampler_weights = get_sampler_weights(train_labels)
            sampler = WeightedRandomSampler(sampler_weights, len(train_dataset), replacement=True)

            # create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with tqdm(total=num_epochs, desc='epoch', position=1, leave=False) as batch_pbar:
            for epoch in range(start_epoch, start_epoch+num_epochs):
                # reset loss, labels and predictions to compute epoch metrics
                total_train_loss = 0
                total_test_loss = 0

                y_true_train = []
                y_score_train = []

                y_true_test = []
                y_score_test = []

                # train loop
                model.train()
                optimizer.zero_grad()
                i = 0
                iters_to_accumulate = 32
                for ct_batch, pet_batch, labels_batch in tqdm(train_loader, position=2, desc='train batch'):
                    ct_batch = ct_batch.to(device)
                    pet_batch = pet_batch.to(device)

                    labels_batch = torch.squeeze(labels_batch).to(device)

                    outputs = model(ct_batch, pet_batch)

                    #loss = criterion(torch.squeeze(outputs[0]), labels_batch) / iters_to_accumulate

                    loss = criterion(torch.squeeze(outputs[0]), 
                                     torch.squeeze(outputs[2]),
                                     torch.squeeze(outputs[3]),
                                     labels_batch) / iters_to_accumulate

                    y_true, y_score = get_y_true_and_pred(y_true=labels_batch, y_pred=outputs[0], cpu=True)

                    y_true_train.append(y_true)
                    y_score_train.append(y_score)

                    total_train_loss += loss.item() * iters_to_accumulate

                    loss.backward()

                    if (i + 1) % iters_to_accumulate == 0 or i + 1 == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                    i += 1
                # test loop
                model.eval()
                with torch.no_grad():
                    for ct_batch, pet_batch, labels_batch in tqdm(test_loader, position=2, desc='test batch'):
                        ct_batch = ct_batch.to(device)
                        pet_batch = pet_batch.to(device)
                        labels_batch = torch.squeeze(labels_batch).to(device)

                        outputs = model(ct_batch, pet_batch)

                        #loss = criterion(torch.squeeze(outputs[0]), labels_batch)
                        loss = criterion(torch.squeeze(outputs[0]),
                                         torch.squeeze(outputs[2]),
                                         torch.squeeze(outputs[3]),
                                         labels_batch)

                        y_true, y_score = get_y_true_and_pred(y_true=labels_batch, y_pred=outputs[0], cpu=True)

                        y_true_test.append(y_true)
                        y_score_test.append(y_score)

                        total_test_loss += loss.item()

                scheduler.step()
                avg_train_loss = total_train_loss / len(train_loader)
                avg_test_loss = total_test_loss / len(test_loader)

                batch_pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})
                batch_pbar.update()

                # generate y_true and y_pred for each split in the epoch
                y_true_train = np.concatenate(y_true_train, axis=0)
                y_score_train = np.concatenate(y_score_train, axis=0)
                y_score_train = y_score_train[:, 1]
                y_pred_train = (y_score_train >= 0.5)*1

                y_true_test = np.concatenate(y_true_test, axis=0)
                y_score_test = np.concatenate(y_score_test, axis=0)
                y_score_test = y_score_test[:, 1]
                y_pred_test = (y_score_test >= 0.5)*1

                # create a clasification report of each split
                roc_auc_test = roc_auc_score(y_true_test, y_score_test)
                roc_auc_train = roc_auc_score(y_true_train, y_score_train)

                train_report = classification_report(y_true_train, y_pred_train, output_dict=True, zero_division=0)
                train_report['ROC AUC'] = roc_auc_train
                train_report['kfold'] = kfold
                train_report['loss'] = avg_train_loss
                train_report['epoch'] = epoch
                train_report['split'] = 'train'

                test_report = classification_report(y_true_test, y_pred_test, output_dict=True, zero_division=0)
                test_report['ROC AUC'] = roc_auc_test
                test_report['kfold'] = kfold
                test_report['loss'] = avg_test_loss
                test_report['epoch'] = epoch
                test_report['split'] = 'test'

                train_report_str = print_classification_report(train_report)
                test_report_str = print_classification_report(test_report)

                # save train and test clasification reports into a json file
                with open(os.path.join(save_dir, f'train_metrics_{epoch}.json'), 'w') as file:
                    json.dump(train_report, file)

                with open(os.path.join(save_dir, f'test_metrics_{epoch}.json'), 'w') as file:
                    json.dump(test_report, file)

                # save a plot of the train an test loss
                train_metrics['kfold'].append(kfold)  # TODO: plot train and test report instead just loss values
                train_metrics['epoch'].append(epoch)
                train_metrics['train_loss'].append(avg_train_loss)
                train_metrics['test_loss'].append(avg_test_loss)
                train_metrics['train_auc'].append(roc_auc_train)
                train_metrics['test_auc'].append(roc_auc_test)
                train_metrics['train_f1'].append(train_report['macro avg']['f1-score'])
                train_metrics['test_f1'].append(test_report['macro avg']['f1-score'])
                train_metrics['train_report'].append(train_report_str.replace('\n', '<br>').replace(' ', '  '))
                train_metrics['test_report'].append(test_report_str.replace('\n', '<br>').replace(' ', '  '))

                df_loss = pd.DataFrame(train_metrics)
                df_loss = df_loss[df_loss['kfold'] == kfold]

                # early stoping
                patience = 15

                df_loss['target_metric'] = df_loss['test_auc'] * np.sqrt(df_loss['test_auc'] * df_loss['train_auc']) * np.sqrt(df_loss['test_f1'] * df_loss['train_f1'])
                df_loss['is_improvement'] = df_loss['target_metric'] >= df_loss['target_metric'].max()

                fig = plot_loss_metrics(df_loss, title=f'{arg_dataset} fold {kfold}')
                fig.write_html(os.path.join(save_dir, 'losses.html'))

                epochs_since_improvement = epoch - df_loss.iloc[df_loss['is_improvement'].argmax()]['epoch']

                # save .pth model checkpoint
                if df_loss['target_metric'].iloc[-1] >= df_loss['target_metric'].mean():
                    save_checkpoint(model, save_dir, epoch)

                df_loss['target_metric'] = df_loss['test_auc'] * np.sqrt(df_loss['test_auc'] * df_loss['train_auc'])
                df_loss['is_improvement'] = df_loss['target_metric'] >= df_loss['target_metric'].max()
                epochs_since_improvement = epoch - df_loss.iloc[df_loss['is_improvement'].argmax()]['epoch']

                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
