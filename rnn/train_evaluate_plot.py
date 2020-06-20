import json
import os
import torch
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from .generate_dataloaders import generate_dataset, datasets_max_min, generate_dataloader
from .model import LSTMDrop


class JSONFileNotFound(Exception):
    pass


def clean_max_and_min(model_name):
    """ parse the max and min from the json file """
    full_path = f'./data/json/{model_name}.json'
    if not os.path.isfile(full_path):
        raise JSONFileNotFound('JSON file with max and min not found')
    with open(f'./data/json/{model_name}.json', 'r') as json_file:
        data = json.load(json_file)
    return data['max'], data['min']


def train(train_data_path, test_data_path, model_name, batch_size,
          seq_length, hidden_size, num_layers, dropout, num_epochs,
          learning_rate, weight_decay, step_size, gamma):
    # write max and min to a json file (only have to do this once)
    train_data = datasets_max_min(train_data_path, test_data_path, model_name)
    max_d, min_d = clean_max_and_min(model_name)
    device = torch.device('cuda')
    train_loader = generate_dataloader(train_data, seq_length, max_d, min_d,
                                       batch_size)
    model = LSTMDrop(hidden_size, num_layers, dropout, device).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    print('Start training ...')
    # just to make sure we are in training mode
    model.train()
    for epoch in range(num_epochs):
        training_loss = 0.0
        for batch_id, (train_x, train_y) in enumerate(train_loader):
            train_x, train_y = train_x.to(device), train_y.to(device)
            optimizer.zero_grad()
            outputs = model(train_x)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        scheduler.step()
        print('Epoch Summary --> Epoch: %d, Epoch Loss: %1.8f, LR: %1.8f' % (
               epoch+1, training_loss/len(train_loader), optimizer.param_groups[0]['lr']))
    print('Training complete!')
    torch.save(model.state_dict(), (f'./trained_models/{model_name}.pt'))


def test(test_data_path, model_name, seq_length, hidden_size,
         num_layers, batch_size=1):
    # dropout does not matter here since we are in eval mode
    dropout = 0
    max_d, min_d = clean_max_and_min(model_name)
    test_data = generate_dataset(test_data_path, 'GHI')
    device = torch.device('cuda')
    test_loader = generate_dataloader(test_data, seq_length, max_d, min_d,
                                      batch_size)
    model = LSTMDrop(hidden_size, num_layers, dropout, device).to(device)
    model.load_state_dict(torch.load(f'./trained_models/{model_name}.pt', map_location=device))
    criterion = torch.nn.MSELoss()
    print('Start testing ...')
    model.eval()
    testing_loss = 0.0
    predicted = []
    actual = []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = model(x)
            predicted.append(outputs.item())
            actual.append(y.item())
            loss = torch.sqrt(criterion(outputs, y))
            testing_loss += loss.item()
    # this is the evalutation loss over the whole test set
    print('RMSE: ' + str(testing_loss/len(test_loader)))
    # inverse normalization
    predicted = ((np.array(predicted)+1)/2)*(max_d-min_d)+min_d
    actual = ((np.array(actual)+1)/2)*(max_d-min_d)+min_d
    print('Done testing!')
    dates = generate_dataset(test_data_path, 'Date')
    return dates, predicted, actual


def pretty_plot(dates, predicted, actual, seq_length,
                s_split, e_split, image_name, title='Predicted vs Actual'):
    # more error handling that can be done here
    if (s_split < 0):
        raise ValueError('s_split can not be less than 0')
    # if e_split is longer than predicted
    if (e_split >= len(predicted)):
        e_split = len(predicted) - 1
        print(f'The e_split index exceeds the size of the list defaulting the value \
              to the length of the list -1: {e_split}')
    dates = dates[(seq_length+1+s_split):e_split]
    predicted = predicted[s_split:e_split]
    actual = actual[s_split:e_split]
    print('Creating plot ..')
    with plt.style.context('seaborn'):
        _, ax = plt.subplots(figsize=(20,6))
        dates, predicted, actual = zip(*sorted(zip(dates, predicted, actual),
                                               key=lambda date: date[0]))
        ax.plot(dates, predicted, alpha=0.7, linewidth=3, label='Predicted')
        ax.plot(dates, actual, alpha=0.7, linewidth=3, label='Actual')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_title(title, fontsize=20)
        ax.set_xlabel('Date Time')
        ax.set_ylabel('GHI (w/m^2)')
        ax.legend()
        image_path = f'./plots/{image_name}.png'
        print(f'Saving image to the path: {image_path}')
        plt.savefig(image_path)
        print('Done plotting!')
