# ViT-Deep-Radiomics

## Dataset
To get started, download the [CSM and Stanford Radiogenomics PET-CT datasets](https://drive.google.com/drive/folders/12v969-5JwiREUnyZno69H0iVOkdgSjqj?usp=sharing) and place them in the `data/lung_radiomics/` directory as follows:
```
data/lung_radiomics/
├── lung_radiomics_datasets.csv
└── lung_radiomics_datasets.hdf5
```

## Precalculate Patch Embeddings
To obtain patch tokens for each segmented slice of a patient, use the `tfds_dense_descriptor.py` script. Each modality (PET and CT) needs to be processed separately. The available backbones are [MedSAM](https://github.com/bowang-lab/MedSAM), [DinoV2](https://github.com/facebookresearch/dinov2) & [SMDino](https://github.com/larosi/steerable-medical-dino/tree/main)

Run the following commands:

For CT:
```bash
python tfds_dense_descriptor.py --model_name 'medsam' --model_path "model.pth" --modality 'ct'
```

For PET:
```bash
python tfds_dense_descriptor.py --model_name 'medsam' --model_path "model.pth" --modality 'pet'
```

## Create Metadata DataFrame
Once the patch embeddings are calculated, create a dataframe with metadata by running:
```bash
python merge_dataframe_features.py
```

## Split Patients using K-Fold
Use the following command to split patients into training and testing sets based on K-fold cross-validation. A `.yaml` file will be generated containing `patient_ids` for each fold and modality.
```bash
python split_patients.py
```

## Train Models
Model training is based on precalculated patch embeddings. Training parameters such as the number of layers and learning rate can be found in `conf/parameters_models.yaml`, while the model architectures are defined in `src/model_archs.py`. Use the `train_models.py` script to train the models:

```bash
cd ./src/
python ./train_models.py --arch "transformer" --dataset "stanford" --modality "petct" --gpu 0 --loss "crossmodal"
python ./train_models.py --arch "transformer" --dataset "santa_maria" --modality "petct" --gpu 1 --loss "crossmodal"
```

## Evaluate Metrics
During training and evaluation, a `.json` file containing relevant metrics is created for each epoch. To aggregate these metrics across folds and modalities, use the `avg_kfold_metrics.py` script, which will generate a `*_metrics_summary.csv` file for comparison:

```bash
python avg_kfold_metrics.py
```