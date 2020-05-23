import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from generate_dataloaders import generate_testloader
from models import LSTMDrop


def test(train_data_path, test_data_path, model_name, s_split, e_split,
         seq_length=64, hidden_size=35, num_layers=2, dropout=0.3):
    device = torch.device('cuda')
    batch_size = 1
    test_loader, test_max, test_min, dates = generate_testloader(train_data_path,
                                             test_data_path, seq_length, batch_size,
                                             s_split, e_split)
    input_size = 1
    model = LSTMDrop(input_size, hidden_size, num_layers, dropout, device).to(device)
    model.load_state_dict(torch.load(('./trained_models/' + model_name + '.pt'), map_location=device))
    model.eval()
    criterion = torch.nn.MSELoss()
    predicted = []
    actual = []
    print('Start testing ...')
    testing_loss = 0.0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = model(x)
            predicted.append(outputs.item())
            actual.append(y.item())
            loss = torch.sqrt(criterion(outputs, y))
            testing_loss += loss.item()
    print('RMSE: ' + str(testing_loss/len(test_loader)))
    predicted = ((np.array(predicted)+1)/2)*(test_max-test_min)+test_min
    actual = ((np.array(actual)+1)/2)*(test_max-test_min)+test_min
    print('Done testing!\n')
    return dates, predicted, actual


def pretty_plot(dates, predicted, actual, image_name, title):
    print('Creating plot ..')
    with plt.style.context('seaborn'):
        _, ax = plt.subplots(figsize=(20,6))
        dates, predicted, actual = zip(*sorted(zip(dates, predicted, actual), key=lambda date: date[0]))
        ax.plot(dates, predicted, alpha=0.7, linewidth=3, label='Predicted')
        ax.plot(dates, actual, alpha=0.7, linewidth=3, label='Actual')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_title(title, fontsize=20)
        ax.set_xlabel('Date Time')
        ax.set_ylabel('GHI (w/m^2)')
        ax.legend()
        image_path = '../../images/' + image_name + '.png'
        print('Saving image to the path: ' + image_path)
        plt.savefig(image_path)
        print('Done plotting!')
