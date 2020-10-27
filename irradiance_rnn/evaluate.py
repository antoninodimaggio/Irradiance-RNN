import torch

from .generate_dataloader import dataloader, inverse_norm, norm_use_scaler
from .model import LSTMDrop


def evaluate(
    lat, lon, test_years, seq_length, model_name, start_date, end_date,
    hidden_size, num_layers, **kwargs
):
    DROPOUT, BATCH_SIZE = 0, 1
    dates, test_loader = dataloader(
        lat, lon, test_years, seq_length, BATCH_SIZE, model_name,
        norm_use_scaler, start_date, end_date
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMDrop(hidden_size, num_layers, DROPOUT, device).to(device)
    model.load_state_dict(
        torch.load(f'./objects/trained_models/{model_name}.pt', map_location=device)
    )
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
    rmse = testing_loss/len(test_loader)
    print(f'RMSE: {rmse}')
    print('Done testing!')
    return (
        dates[seq_length+1:],
        inverse_norm(predicted, model_name),
        inverse_norm(actual, model_name),
        rmse
    )
