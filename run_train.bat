cd .\src\
python .\train_petct_conv.py --arch "conv" --dataset "stanford" --modality "ct"
python .\train_petct_conv.py --arch "conv" --dataset "santa_maria" --modality "ct"
python .\train_petct_conv.py --arch "transformer" --dataset "stanford" --modality "ct"
python .\train_petct_conv.py --arch "transformer" --dataset "santa_maria" --modality "ct"
python .\train_petct_conv.py --arch "conv" --dataset "stanford" --modality "pet"
python .\train_petct_conv.py --arch "conv" --dataset "santa_maria" --modality "pet"
python .\train_petct_conv.py --arch "transformer" --dataset "stanford" --modality "pet"
python .\train_petct_conv.py --arch "transformer" --dataset "santa_maria" --modality "pet"