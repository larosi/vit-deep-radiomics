# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:38:14 2023

@author: Mico
"""


import os
import numpy as np

from skimage.transform import resize
from scipy.ndimage import rotate
from skimage.color import gray2rgb
import torch
import tensorflow_datasets as tfds
import nibabel as nib
from segment_anything import sam_model_registry
from tqdm import tqdm
import pandas as pd
import h5py

from visualization_utils import (crop_image,
                                 extract_coords,
                                 extract_roi,
                                 visualize_features,
                                 hu_to_rgb_vectorized)


def prepare_image(img):
    if len(img.shape) < 3:
        img = gray2rgb(img)
        img_tensor = resize(img, (1024, 1024))
    else:
        img_tensor = resize(img, (896, 896))
    img_tensor = img_tensor.transpose((2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = torch.as_tensor(img_tensor, dtype=torch.float32).cuda()
    return img_tensor

def load_model(model_name, model_path=None):
    if model_name == 'dinov2':
        model = load_dinov2()
    elif model_name == 'medsam':
        model = load_medsam(model_path)
    model.model_name = model_name
    return model


def load_dinov2(backbone_size='small'):
    backbone_archs = {"small": "vits14",
                      "base": "vitb14",
                      "large": "vitl14",
                      "giant": "vitg14"}

    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    model.eval()
    model.cuda()
    return model

def load_medsam(model_path=None):
    if model_path is None:
        model_path = os.path.join('models', 'backbones', 'medsam_vit_b.pth')
    device = torch.cuda.current_device()
    model = sam_model_registry['vit_b'](model_path)
    model = model.to(device)
    model.eval()
    return model


def get_dense_descriptor(model, img):
    img_tensor = prepare_image(img)
    if model.model_name == 'medsam':
        features_tensor = model.image_encoder(img_tensor)
        features = features_tensor.cpu().detach().numpy()
        features = np.squeeze(features)
        features = np.transpose(features, (1, 2, 0))
    else:
        features_tensor = model.patch_embed(img_tensor)
        features = features_tensor.cpu().detach().numpy()
        features = np.squeeze(features)

        featmap_size = int(np.sqrt(features.shape[0]))
        features = features.reshape(featmap_size, featmap_size, features.shape[1])

    del img_tensor
    del features_tensor
    torch.cuda.empty_cache()

    return features


def save_features(filename, all_features, all_masks, patient_id):
    with h5py.File(filename, 'a') as h5f:
        if patient_id in h5f:
            print(f'features for {patient_id} already exists')
            del h5f[patient_id] 
        patient_group = h5f.create_group(patient_id)
        for i, (feature, mask) in enumerate(zip(all_features, all_masks)):
            patient_group.create_dataset(f'features/{i}', data=feature)
            patient_group.create_dataset(f'masks/{i}', data=mask)


def tfds2voxels(ds, patient_id, pet=False):
    eps = np.finfo(np.float32).eps
    img = []
    mask = []
    label = []
    for sample in ds[patient_id]:
        pet_liver_mean = 1
        if pet:
            pet_liver = sample['pet_liver'].numpy()
            pet_liver_mean = pet_liver[pet_liver != 0].mean() + 1e-10
        img += [sample['img_exam'].numpy() / pet_liver_mean]
        mask += [sample['mask_exam'].numpy()]
        label += [sample['egfr_label'].numpy()]
    img = np.dstack(img)
    mask = np.dstack(mask)
    spatial_res = sample['exam_metadata']['space_directions'].numpy()
    return img, mask, label, spatial_res


def windowing_ct(width, level):
    """
    chest
    - lungs W:1500 L:-600
    - mediastinum W:350 L:50

    abdomen
    - soft tissues W:400 L:50
    - liver W:150 L:30

    spine
    - soft tissues W:250 L:50
    - bone W:1800 L:400

    head and neck
    - brain W:80 L:40
    - subdural W:130-300 L:50-100
    - stroke W:8 L:32 or W:40 L:40
    - temporal bones W:2800 L:600 or W:4000 L:700
    - soft tissues: W:350–400 L:20–60

    source: https://radiopaedia.org/articles/windowing-ct
    """
    lower_bound = level - width/2
    upper_bound = level + width/2
    return lower_bound, upper_bound


def generate_features(model, img_3d, mask_3d, tqdm_text, display=False):
    bigger_mask = np.sum(mask_3d, axis=-1) > 0

    h, w = bigger_mask.shape
    xmin, ymin, xmax, ymax = extract_coords(bigger_mask, margin=2)
    crop_size = max(xmax-xmin, ymax-ymin)*2
    xmid, ymid = int(xmin + (xmax-xmin)/2), int(ymin + (ymax-ymin)/2)
    xmin, ymin, xmax, ymax = xmid-crop_size, ymid-crop_size, xmid+crop_size, ymid+crop_size

    img_3d = crop_image(img_3d, xmin, ymin, xmax, ymax)
    mask_3d = crop_image(mask_3d, xmin, ymin, xmax, ymax)
    bigger_mask = crop_image(bigger_mask, xmin, ymin, xmax, ymax)

    features_list = []
    mask_list = []
    for slice_i in tqdm(range(0, img_3d.shape[2]), desc=tqdm_text, leave=False):
        if model.model_name == 'medsam':
            img = img_3d[:, :, slice_i]
        else:
            img = img_3d[:, :, slice_i, :]
        mask = mask_3d[:, :, slice_i] > 0
        features = get_dense_descriptor(model, img)
        crop_features = extract_roi(features, bigger_mask)
        crop_mask = extract_roi(mask, bigger_mask)
        features_list.append(crop_features)
        mask_list.append(crop_mask)
        if display:
            visualize_features(img, features, mask)
    return features_list, mask_list


def apply_window_ct(ct, width, level):
    ct_min_val, ct_max_val = windowing_ct(width, level)
    ct_range = ct_max_val - ct_min_val
    ct = (ct - ct_min_val) / ct_range
    ct = np.clip(ct, 0, 1)
    return ct


def flip_image(image, mask, flip_type):
    image_flip = image.copy()
    mask_flip = mask.copy()
    if flip_type == 'horizontal':
        return image_flip[:, ::-1, ...], mask_flip[:, ::-1, ...]
    elif flip_type == 'vertical':
        return image_flip[::-1, ...], mask_flip[::-1, ...]
    return image_flip, mask_flip


def rotate_image(image, mask, angle, axes=(0, 1)):
    image_rot = image.copy()
    mask_rot = mask.copy()
    if angle == 0:
        return image_rot, mask_rot
    image_rot = rotate(image_rot, angle, axes=axes, reshape=False, mode='nearest')
    image_rot = np.clip(image_rot, 0, 1)
    mask_rot = rotate(mask_rot, angle, axes=axes, reshape=False, mode='nearest')
    mask_rot = mask_rot > 0
    return image_rot, mask_rot


if __name__ == "__main__":
    model_name = 'medsam'
    model_path = os.path.join('..', 'models', 'backbones', 'medsam', 'medsam_vit_b.pth')
    dataset_path = r'D:\datasets\medseg\lung_radiomics'

    model = load_model(model_name, model_path)

    datasets = ['santa_maria_dataset', 'stanford_dataset']

    dataframes = []
    for dataset_name in datasets:
        feature_folder = r'..\data\features'
        features_dir = os.path.join(feature_folder, dataset_name)
        os.makedirs(features_dir, exist_ok=True)
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
            for modality in ['ct']:
                df_path = os.path.join(features_dir, f'{patient_id}_{modality}.parquet')
                features_file = os.path.join(feature_folder, f'features_masks_{modality}.hdf5')
                if not os.path.exists(df_path):
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

                        # normalize pixel values
                        if modality == 'ct':
                            if model.model_name == 'medsam':
                                img_raw = apply_window_ct(img_raw, width=800, level=40)
                            else:
                                img_raw = hu_to_rgb_vectorized(img_raw) / 255.0
                        else:
                            pass  # TODO: define a preprocess for PET images

                        # extract patch features of each slice
                        df = {'slice': [],
                              'angle': [],
                              'flip': []}

                        all_features = []
                        all_masks = []
                        angles = []
                        flips = []
                        slices = []
                        # apply flip and rotation to use them as offline data augmentation
                        for flip_type in [None, 'horizontal', 'vertical']:
                            image_flip, mask_flip = flip_image(img_raw, mask_raw, flip_type)
                            for angle in range(0, 180, 45):
                                image, mask = rotate_image(image_flip, mask_flip, angle)
                                features, features_mask = generate_features(model=model,
                                                                            img_3d=image,
                                                                            mask_3d=mask,
                                                                            tqdm_text=f'{modality} {patient_id}',
                                                                            display=False)

                                all_masks += features_mask
                                all_features += features

                                df['angle'] += [angle] * len(features)
                                df['flip'] += [flip_type] * len(features)
                                df['slice'] += list(range(0, len(features)))

                        # store metadata of each featuremap in a dataframe
                        df = pd.DataFrame(df)
                        df.reset_index(drop=False, inplace=True)
                        df = df.rename(columns={'index': 'feature_id'})
                        df['patient_id'] = patient_id
                        df['label'] = label
                        df['dataset'] = dataset_name.replace('_dataset', '')
                        df['modality'] = modality
                        df['augmentation'] = np.logical_not(np.logical_and(df['flip'] is None,  df['angle'] == 0))
                        df['spatial_res'] = [spatial_res] * df.shape[0]
                        df.to_parquet(df_path)
                        save_features(features_file, all_features, all_masks, patient_id)
