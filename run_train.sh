#!/bin/bash

cd ./src/
python ./train_models.py --arch "conv" --dataset "stanford" --modality "pet" --gpu 1
python ./train_models.py --arch "conv" --dataset "stanford" --modality "ct" --gpu 1
python ./train_models.py --arch "conv" --dataset "santa_maria" --modality "pet" --gpu 1
python ./train_models.py --arch "conv" --dataset "santa_maria" --modality "ct" --gpu 1
python ./train_models.py --arch "transformer" --dataset "stanford" --modality "pet" --gpu 1
python ./train_models.py --arch "transformer" --dataset "stanford" --modality "ct" --gpu 1
python ./train_models.py --arch "transformer" --dataset "santa_maria" --modality "pet" --gpu 1
python ./train_models.py --arch "transformer" --dataset "santa_maria" --modality "ct" --gpu 1
python ./train_models.py --arch "transformer" --dataset "stanford" --modality "petct" --gpu 1
python ./train_models.py --arch "transformer" --dataset "santa_maria" --modality "petct" --gpu 1