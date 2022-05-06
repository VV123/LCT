# Modeling Spatio-Temporal Dynamics using Local Contextual Transformer

## Requirements

- Python 3
- PyTorch (>= 1.0)
- numpy
- shapely
- pytorch_forecasting

## Data

Store [data file](https://drive.google.com/drive/folders/110UJnpo1ZVNkrOM7wuQ90ahVJZatZECF?usp=sharing) in the root directory.

## How to run

  
```
python3 main.py --epoch 1000  \
        --lookback 24 --path model.h5 \
        --layer 3 --size 3 \
        --loaddata —-mode infer
```

- ```size```: K-hop neighbors
- ```layer```: # of Transformer encoder layers
