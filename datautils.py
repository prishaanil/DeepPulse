

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange', 'PTB-XL',
        ]

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    if params.dset == 'ettm1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'ettm2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'etth1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'etth2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    

    elif params.dset == 'electricity':
        root_path = '/data/datasets/public/electricity/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'traffic':
        root_path = '/data/datasets/public/traffic/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 'weather':
        root_path = '/data/datasets/public/weather/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'illness':
        root_path = '/data/datasets/public/illness/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'exchange':
        root_path = '/data/datasets/public/exchange_rate/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
        
    elif params.dset == 'PTB-XL':
        """
        Creates DataLoaders for training and validation.
        """
        assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
        if not hasattr(params, 'use_time_features'):
            params.use_time_features = False

        if params.dset == 'PTB-XL':
            """
            Creates DataLoaders for training and validation.
            """
            # Initialize DataLoaders
            dls = DataLoaders(
                datasetCls=Dataset_PTB_XL,
                dataset_kwargs={
                    'root_path': params.root_path,
                    'data_path': params.data_path,
                    'target': params.target,
                    'scale': (params.scaler == 'standard'),
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
            )

            # Extract the number of variables (channels) and sequence length
            sample_data, sample_target = dls.train.dataset[0]  # Get a single sample
            if len(sample_data.shape) == 2:  # If the data is 2D
                num_channels = sample_data.shape[1]  # Number of channels
                seq_len = sample_data.shape[0]  # Sequence length
            else:  # If the data is 1D
                num_channels = 1  # Assume single channel
                seq_len = sample_data.shape[0]  # Sequence length

            print(f"Number of Channels: {num_channels}, Sequence Length: {seq_len}")
            print(f"Target vector shape: {sample_target.shape}")

            # Return DataLoaders and metadata
            return dls, num_channels, seq_len


    # dataset is assume to have dimension len x nvars



if __name__ == "__main__":
    class Params:
        dset= 'PTB-XL'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='S'
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()