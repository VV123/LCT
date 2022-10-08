# LCT
## Requirements

- Python 3
- PyTorch (>= 1.0)
- numpy
- shapely
- pytorch_forecasting

## Data

[Raw data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (working with Parquet Format [Link](https://www1.nyc.gov/assets/tlc/downloads/pdf/working_parquet_format.pdf))

Store [data file]() in the root directory.

## How to run

  
```
python3 main.py --epoch 1000  \
        --lookback 24 --path model.h5 \
        --layer 3 --size 3 \
        --loaddata —-mode train
```

- ```size```: K-hop neighbors
- ```layer```: # of Transformer encoder layers


The full implementation and documents will be disclosed later.
