import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class CSVFileNotFound(Exception):

    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

    def __str__(self):
        return f'The path {self.filename} does not exist'


# TODO: what happens when that date does not exist
def reshape_data(df, start_date, end_date):
    if start_date and end_date:
        df = df.set_index('Date')[start_date:end_date]
    elif start_date:
        df = df.set_index('Date')[start_date:]
    elif end_date:
        df = df.set_index('Date')[:end_date]
    return df.index, df.loc[:, 'GHI'].values.reshape(-1, 1)


def norm_save_scaler(df, model_name, start_date, end_date):
    dates, data = reshape_data(df, start_date, end_date)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    joblib.dump(scaler, f'./objects/scalers/{model_name}.gz')
    return dates, scaled_data


def norm_use_scaler(df, model_name, start_date, end_date):
    dates, data = reshape_data(df, start_date, end_date)
    scaler = joblib.load(f'./objects/scalers/{model_name}.gz')
    return dates, scaler.transform(data)


def inverse_norm(data, model_name):
    scaler = joblib.load(f'./objects/scalers/{model_name}.gz')
    return scaler.inverse_transform(np.array(data).reshape(-1, 1))


def concat_csv(lat, lon, years):
    df = pd.DataFrame(columns=['Date', 'GHI'])
    for year in years.split(','):
        path = f'./data/csv/{lat}_{lon}/{year}.csv'
        if not os.path.isfile(path):
            raise CSVFileNotFound(path)
        df = df.append(
            pd.read_csv(f'./data/csv/{lat}_{lon}/{year}.csv', parse_dates=True),
            ignore_index=True
        )
    return df


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def create_dataloader(scaled_data, seq_length, batch_size):
    x, y = create_sequences(scaled_data, seq_length)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float()
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True
    )
    return dataloader


def dataloader(
    lat, lon, years, seq_length, batch_size, model_name, norm_callback,
    start_date, end_date
):
    df = concat_csv(lat, lon, years)
    dates, scaled_data = norm_callback(df, model_name, start_date, end_date)
    dataloader = create_dataloader(scaled_data, seq_length, batch_size)
    return dates, dataloader
