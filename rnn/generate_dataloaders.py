import json
import numpy as np
import os
import pandas as pd
import torch


class CSVFileNotFound(Exception):
    pass


def generate_dataset(data_path, col_name):
    # if file does not exist
    if not os.path.isfile(data_path):
        raise CSVFileNotFound('CSV file does not exist or the name is incorrect')
    data = pd.read_csv(data_path, parse_dates=['Date']).loc[:, col_name].values
    return data


def datasets_max_min(train_data_path, test_data_path, model_name):
    """ find the max and min of the whole dataset and write to a json file"""
    train_data = generate_dataset(train_data_path, 'GHI')
    test_data = generate_dataset(test_data_path, 'GHI')
    max_d = int(max(train_data.max(), test_data.max()))
    min_d = int(min(train_data.min(), test_data.min()))
    max_min_dict = {'max': max_d, 'min': min_d}
    with open(f'./data/json/{model_name}.json', 'w') as json_file:
        json.dump(max_min_dict, json_file)
    # might as well return train_data
    return train_data


def generate_dataloader(data, seq_length, max, min,
                        batch_size, max_range=1, min_range=-1):
    """ normalize the data between -1 and 1 and return the dataloader """
    data = (data-min)/(max-min)*(max_range-min_range)+min_range
    x, y = create_sequences(np.expand_dims(data, axis=-1), seq_length)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x).float(),
                                             torch.from_numpy(y).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=2, shuffle=False,
                                             pin_memory=True)
    return dataloader


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
