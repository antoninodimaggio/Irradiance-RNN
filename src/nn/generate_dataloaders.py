import numpy as np
import pandas as pd
import torch


def generate_trainloader(train_data_path, test_data_path, seq_length,
                         batch_size, max_range=1, min_range=-1):
    # both train_data and test_data so that I can scale the data correctly
    train_data = pd.read_csv(train_data_path, parse_dates=['Date']).loc[:, 'GHI'].values
    test_data = pd.read_csv(test_data_path, parse_dates=['Date']).loc[:, 'GHI'].values
    train_max = max(train_data.max(), test_data.max())
    train_min = min(train_data.min(), test_data.min())
    train_data = ((train_data-train_min)/(train_max-train_min))*(max_range - min_range)+min_range
    x, y = create_sequences(np.expand_dims(train_data, axis=-1), seq_length)
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(x).float(),
                                               torch.from_numpy(y).float())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               num_workers=2, shuffle=False,
                                               pin_memory=True)
    return train_loader


def generate_testloader(train_data_path, test_data_path, seq_length,
                        batch_size, s_split, e_split, max_range=1, min_range=-1):
    # both train_data and test_data so that I can scale the data correctly
    train_data = pd.read_csv(train_data_path, parse_dates=['Date']).loc[:, 'GHI'].values
    data = pd.read_csv(test_data_path, parse_dates=['Date'])
    dates = data.loc[:, 'Date'].values[(seq_length+1+s_split):e_split]
    test_data = data.loc[:, 'GHI'].values
    test_max = max(train_data.max(), test_data.max())
    test_min = min(train_data.min(), test_data.min())
    test_data = ((test_data-test_min)/(test_max-test_min))*(max_range - min_range)+min_range
    x, y = create_sequences(np.expand_dims(test_data, axis=-1), seq_length)
    test_set = torch.utils.data.TensorDataset(torch.from_numpy(x[s_split:e_split]).float(),
                                              torch.from_numpy(y[s_split:e_split]).float())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                               num_workers=2, shuffle=False,
                                               pin_memory=True)
    return test_loader, test_max, test_min, dates


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
