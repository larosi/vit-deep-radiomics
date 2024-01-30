# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:22:10 2024

@author: Mico
"""

import os
import pandas as pd
import numpy as np
import json
import h5py
from tqdm import tqdm
import plotly.graph_objs as go
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class TransformerNoduleClassifier(nn.Module):
    def __init__(self, input_dim, dim_feedforward, num_heads, num_classes, num_layers,):
        super(TransformerNoduleClassifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   dim_feedforward=dim_feedforward,
                                                   nhead=num_heads,
                                                   activation="gelu",
                                                   batch_first=True,
                                                   dropout=0.1)
        self.norm = nn.LayerNorm(input_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.classifier = MLPLayer(input_dim, input_dim*2, num_classes)

    def forward(self, x):
        batch, seq_len, feature_dim = x.shape
        cls_token = self.cls_token.repeat(batch, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.norm(x)
        x = self.transformer_encoder(x)
        return self.classifier(x[:,0,:]), x[:,0,:]

    def save_checkpoint(self, save_dir, epoch):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        epoch_str = str(epoch).zfill(4)
        model_path = os.path.join(save_dir, f'model_epoch_{epoch_str}.pth')
        self.save(model_path)

    def load_checkpoint(self, save_dir, epoch):
        epoch_str = str(epoch).zfill(4)
        model_path = os.path.join(save_dir, f'model_epoch_{epoch_str}.pth')
        self.load(model_path)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(model_path, map_location=device))

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_features, out_features, dropout_rate=0.1):
        super(MLPLayer, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_features, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class NoduleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, div=2):
        super(NoduleClassifier, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_dim, out_channels=input_dim//div, kernel_size=3, padding=1)
        self.se1 = SELayer(input_dim//div)
        self.conv2 = nn.Conv3d(in_channels=input_dim//div, out_channels=input_dim//(div*div), kernel_size=3, padding=1)
        self.se2 = SELayer(input_dim//(div*div))

        self.fc1 = nn.Linear(input_dim//(div*div), input_dim)
        self.classifier = MLPLayer(input_dim, input_dim*2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.se1(x)
        x = F.relu(self.conv2(x))
        x = self.se2(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        return self.classifier(x), x

    def save_checkpoint(self, save_dir, epoch):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        epoch_str = str(epoch).zfill(4)
        model_path = os.path.join(save_dir, f'model_epoch_{epoch_str}.pth')
        self.save(model_path)

    def load_checkpoint(self, save_dir, epoch):
        epoch_str = str(epoch).zfill(4)
        model_path = os.path.join(save_dir, f'model_epoch_{epoch_str}.pth')
        self.load(model_path)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(model_path, map_location=device))


def positional_encoding_3d(x, y, z, D, scale=10):
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    n_points = x.shape[0]
    encoding = np.zeros((n_points, D))

    for i in range(D // 6):
        encoding[:, 2*i] = np.sin(x / (scale ** (6 * i / D)))
        encoding[:, 2*i + 1] = np.cos(x / (scale ** (6 * i / D)))
        encoding[:, 2*i + D // 3] = np.sin(y / (scale ** (6 * i / D)))
        encoding[:, 2*i + 1 + D // 3] = np.cos(y / (scale ** (6 * i / D)))
        encoding[:, 2*i + 2 * D // 3] = np.sin(z / (scale ** (6 * i / D)))
        encoding[:, 2*i + 1 + 2 * D // 3] = np.cos(z / (scale ** (6 * i / D)))

    return encoding


class PETCTDataset3D(Dataset):
    def __init__(self, dataframe, label_encoder, hdf5_path, use_augmentation=False, feature_dim=256, arch='conv'):
        self.dataframe = dataframe.groupby(['patient_id_new'])[['modality', 'dataset', 'label', 'patient_id']].first()
        self.dataframe.reset_index(inplace=True, drop=False)

        self.use_augmentation = use_augmentation

        self.flip_angles = dataframe.groupby(['flip', 'angle'], as_index=False).size()[['flip', 'angle']]

        self.dataframe_aug = dataframe.set_index(['patient_id_new', 'angle', 'flip'])
        self.dataframe_aug = self.dataframe_aug.sort_index()

        self.hdf5_path = hdf5_path
        self.label_encoder = label_encoder
        self.feature_dim = feature_dim
        self.arch = arch

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        patient_id_rew = sample.patient_id_new
        patient_id = sample.patient_id
        label = sample.label

        if self.use_augmentation:
            [[flip, angle]] = self.flip_angles.sample(n=1).values
        else:
            flip = 'None'
            angle = 0

        features = self._get_features(patient_id, patient_id_rew, angle, flip)
        features = torch.as_tensor(features, dtype=torch.float32)

        labels = np.array(label)
        labels = np.expand_dims(labels, axis=-1)
        labels = self.label_encoder.transform(labels.reshape(-1, 1)).toarray()
        labels = torch.as_tensor(labels, dtype=torch.float32)

        return features, labels

    def _get_features(self, patient_id, patient_id_rew, angle, flip):
        feature_ids = self.dataframe_aug.loc[(patient_id_rew, angle, flip)]['feature_id'].values
        features = []
        masks = []

        with h5py.File(self.hdf5_path, 'r') as h5f:
            for feature_id in feature_ids:
                slice_features = h5f[f'{patient_id}/features/{feature_id}'][()]
                slice_mask_orig = h5f[f'{patient_id}/masks/{feature_id}'][()]
                slice_mask = resize(slice_mask_orig, slice_features.shape[0:2])
                slice_mask = np.expand_dims(slice_mask, axis=-1)
                features.append(slice_features*slice_mask)  # elementwise prod feature-mask
                masks.append(slice_mask)

        features = np.transpose(np.stack(features, axis=0), axes=(3, 0, 1, 2))  # (slice, h, w, feat_dim) -> (feat_dim, slice, h, w)
        if self.arch == 'transformer':
            masks = np.transpose(np.stack(masks, axis=0), axes=(1, 2, 0, 3))  # (slice, h, w, 1) -> (h, w, slice, 1)
            h_orig, w_orig = slice_mask_orig.shape[0:2]
            features = np.transpose(np.stack(features, axis=0), axes=(2, 3, 1, 0))  # (h, w, slice, feat_dim)
            h_new, w_new = features.shape[2], features.shape[3]
            spatial_res = self.dataframe_aug.loc[(patient_id_rew, angle, flip)]['spatial_res'].values[0]
            x, y, z = np.meshgrid(np.arange(0, features.shape[0]),
                                  np.arange(0, features.shape[1]),
                                  np.arange(0, features.shape[2]))
            x = (x.flatten()/w_new) * w_orig * spatial_res[0]
            y = (y.flatten()/h_new).flatten() * h_orig * spatial_res[1]
            z = (z.flatten()).flatten() * spatial_res[2]

            x = x - x.mean() + np.random.random(1) * 5 - 2.5
            y = y - y.mean() + np.random.random(1) * 5 - 2.5
            z = z - z.mean() + np.random.random(1) * 5 - 2.5

            pe = positional_encoding_3d(x, y, z, D=self.feature_dim, scale=1000)
            features = features.reshape(-1, self.feature_dim) + pe / 2 # (seq_len, feat_dim)
            features = features[masks.flatten(), :]  # (seq_len, feat_dim)
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
    support = df.loc['macro avg'][-1]

    for row in global_metrics:
        metric_val = df.loc[row][-2]
        new_row = [' ']*len(headers)
        new_row[-2] = metric_val
        new_row[-1] = support
        df.loc[row] = new_row

    local_metrics = [col for col in df.T.columns if col not in global_metrics]
    df_local = df.loc[local_metrics]
    df_global = df.loc[global_metrics].T[-2:-1]
    df_global.index = ['   ']

    print(f'\n{df_global}\n\n{df_local}\n\n')


def plot_loss_metrics(df_loss):
    """ Plot loss vs epoch with the standard desviation between kfolds

    Args:
        df_loss (pd.dataframe): dataframe with the training loss metrics.

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure.

    """
    avg_loss_per_epoch = df_loss.groupby('epoch').agg({'train_loss': ['mean', 'std'],
                                                       'test_loss': ['mean', 'std']}).reset_index()

    avg_loss_per_epoch_columns = ['epoch']
    all_tasks = ['']

    for task in all_tasks:
        for split in ['train', 'test']:
            avg_loss_per_epoch_columns += [f'{split}_loss{task}_mean']
            avg_loss_per_epoch_columns += [f'{split}_loss{task}_std']
    avg_loss_per_epoch.columns = avg_loss_per_epoch_columns
    fig = go.Figure()

    colors = {'train': 'blue', 'test': 'red'}
    colors_rgba = {'train': 'rgba(68, 68, 128, 0.2)',
                   'test': 'rgba(128, 68, 68, 0.2)'}
    symbols = {'': 'circle',
               '_task': 'x',
               '_subtask': 'star'}

    for task in all_tasks:
        for split in ['train', 'test']:
            fig.add_trace(go.Scatter(
                name=f'{split}{task}',
                x=avg_loss_per_epoch['epoch'],
                y=avg_loss_per_epoch[f'{split}_loss{task}_mean'],
                mode='markers+lines',
                line=dict(color=colors[split]),
                marker=dict(color=colors[split], symbol=symbols[task])
            ))
            for sign in [-1, 1]:
                if sign == -1:
                    name = f'{split}{task} - std'
                else:
                    name = f'{split}{task} + std'
                fig.add_trace(go.Scatter(
                    name=name,
                    x=avg_loss_per_epoch['epoch'],
                    y=avg_loss_per_epoch[f'{split}_loss{task}_mean'] + sign * avg_loss_per_epoch[f'{split}_loss{task}_std'],
                    mode='lines',
                    fillcolor=colors_rgba[split],
                    fill='tonexty',
                    marker=dict(color=colors_rgba[split]),
                    line=dict(width=0),
                    showlegend=False
                ))

    fig.update_layout(
        yaxis_title='Loss',
        title='Train Loss per Epoch with Batch Standard Desviation',
        hovermode="x"
    )
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Convert one-hot encoded targets to class indices
        if len(inputs.shape) == 1:
            inputs = torch.unsqueeze(inputs, 0)
            targets = torch.unsqueeze(targets, 0)

        class_indices = torch.argmax(targets, dim=1)

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, class_indices, reduction='none')
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        if self.alpha is not None:
            if not self.alpha.device == inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            loss = (self.alpha[class_indices] * (1 - pt) ** self.gamma * ce_loss).mean()
        else:
            loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return loss
    
def find_divisor(slice_count, desired_slices=7):
    if slice_count < desired_slices:
        return 1
    else:
        for divisor in range(desired_slices, slice_count + 1):
            if slice_count % divisor == 0 or slice_count % divisor >= desired_slices:
                return divisor
        return 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D transoformer or CNN for lung nodules clasification")
    parser.add_argument("-a", "--arch", type=str, default="transformer",
                        help="'transformer' or 'conv'")
    parser.add_argument("-d", "--dataset", type=str, default="all",
                        help="dataset 'stanford', 'santa_maria' or 'all'")
    parser.add_argument("-b", "--backbone", type=str, default="medsam",
                        help="backbone ViT encoder 'medsam' or 'dinov2'")
    parser.add_argument("-m", "--modality", type=str, default="ct",
                        help="ct or pet")

    args = parser.parse_args()

    arch = args.arch
    backbone = args.backbone
    modality = args.modality
    use_sampler = True
    arg_dataset = args.dataset
    if arg_dataset == 'all':
        desired_datasets = ['stanford', 'santa_maria']
    else:
        desired_datasets = [arg_dataset]
        
    hdf5_path = os.path.join('..', 'data', 'features', f'features_masks_{modality}.hdf5')
    df_path = os.path.join('..', 'data', 'features', 'petct.parquet')
    #models_save_dir = os.path.join('..', 'models', 'petct')  # TODO:  generate an unique experiment ID
    models_save_dir = os.path.join('..', 'models', 'petct', f'{backbone}_{arch}_{arg_dataset}')

    # load dataframe and apply some filter criteria
    df = pd.read_parquet(df_path)
    df= df[df['dataset'].isin(desired_datasets)]
    df = df[df['modality'] == modality]
    df['flip'] = df['flip'].astype(str)
    df.reset_index(drop=True, inplace=True)
    
    df['divisor'] = 1
    slices_per_patient = df.groupby(['patient_id'])['slice', 'divisor'].max()
    slices_per_patient.describe()
    slices_per_patient['divisor'] = slices_per_patient['slice'].apply(find_divisor)
    slices_per_patient = slices_per_patient['divisor'].to_dict()
    df['divisor'] = df['patient_id'].apply(lambda x: slices_per_patient[x])
    
    df['patient_id_new'] = df.apply(lambda row: f"{row['patient_id']}:{np.ceil(row['slice']/row['divisor']).astype(int)}", axis=1)
    df.groupby('patient_id_new')['label'].describe()
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
                     'test_loss': []}

    # use KFold to split patients stratified by label
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    patients_labels = df.groupby('patient_id')['label'].first()
    patients = patients_labels.index.to_list()
    patients_labels = patients_labels.to_list()

    for kfold, (train_indices, test_indices) in tqdm(enumerate(skf.split(patients, patients_labels)), desc='kfold', leave=False, position=0):   
        save_dir = os.path.join(models_save_dir, modality, f'kfold_{kfold}')
        os.makedirs(save_dir, exist_ok=True)
    
        # get patient_ids of each split
        training_patients = [patients[i] for i in train_indices]
        testing_patients = [patients[i] for i in test_indices]
    
        # filter dataframes based on the split patients
        df_train = df[df['patient_id'].isin(training_patients)]
        df_test = df[df['patient_id'].isin(testing_patients)]
    
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
    
        #  TODO: define training parameters in a .yaml file
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
    
        else:
            batch_size = 1  # TODO: add support for bigger batches using zero padding to create batches of the same size
            start_epoch = 0
            num_epochs = 25
            learning_rate = 0.0001
            num_layers = 1
    
            if backbone == 'medsam':
                feature_dim = 256
                div = 2  # reduction factor of the conv layers
    
                # FIXME: deprecated transformer params
                num_heads = 16
                dim_feedforward = feature_dim*2
    
            else:  # dinov2
                feature_dim = 384
                div = 3  # reduction factor of the conv layers
                # FIXME: deprecated transformer params
                num_heads = 12
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

        print(model)
        model = model.to(device)

        # CrossEntropyLoss because the last layer has one output per class (mutant, wildtype)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

        # create datasets
        train_dataset = PETCTDataset3D(df_train,
                                       label_encoder=EGFR_encoder,
                                       hdf5_path=hdf5_path,
                                       use_augmentation=True,
                                       feature_dim=feature_dim,
                                       arch=arch)

        test_dataset = PETCTDataset3D(df_test,
                                      label_encoder=EGFR_encoder,
                                      hdf5_path=hdf5_path,
                                      use_augmentation=False,
                                      feature_dim=feature_dim,
                                      arch=arch)

        # create a sampler to balance training classes proportion
        if use_sampler:
            train_labels = np.array(list(train_dataset.dataframe.label.values))
            sampler_weights = get_sampler_weights(train_labels)
            sampler = WeightedRandomSampler(sampler_weights, len(train_dataset), replacement=True)

            test_labels = np.array(list(test_dataset.dataframe.label.values))
            test_sampler_weights = get_sampler_weights(test_labels)
            sample_test = WeightedRandomSampler(test_sampler_weights, len(test_dataset)*2, replacement=True)
            # create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=sample_test)
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
                iters_to_accumulate = 1
                for features_batch, labels_batch in tqdm(train_loader, position=2, desc='train batch'):
                    features_batch = features_batch.to(device)

                    labels_batch = torch.squeeze(labels_batch).to(device)

                    outputs = model(features_batch)

                    loss = criterion(torch.squeeze(outputs[0]), labels_batch) / iters_to_accumulate
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
                    for features_batch, labels_batch in tqdm(test_loader, position=2, desc='test batch'):
                        features_batch = features_batch.to(device)
                        labels_batch = torch.squeeze(labels_batch).to(device)

                        outputs = model(features_batch)
                        loss = criterion(torch.squeeze(outputs[0]), labels_batch)

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
                y_pred_train = (y_score_train > 0.5)*1

                y_true_test = np.concatenate(y_true_test, axis=0)
                y_score_test = np.concatenate(y_score_test, axis=0)
                y_score_test = y_score_test[:, 1]
                y_pred_test = (y_score_test > 0.5)*1

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

                print_classification_report(train_report)
                print_classification_report(test_report)

                # save .pth model checkpoint
                model.save_checkpoint(save_dir, epoch)

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
                df_loss = pd.DataFrame(train_metrics)
                df_loss = df_loss[df_loss['kfold'] == kfold]

                fig = plot_loss_metrics(df_loss)
                fig.write_html(os.path.join(save_dir, 'losses.html'))

                # early stoping
                patience = 8
                df_loss['is_improvement'] = df_loss['test_loss'] <= df_loss['test_loss'].cummin()
                df_loss['epochs_since_improvement'] = (~df_loss['is_improvement']).cumsum()

                if df_loss['epochs_since_improvement'].iloc[-1] >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
