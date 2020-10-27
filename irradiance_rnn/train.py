import torch

from .generate_dataloader import dataloader, norm_save_scaler
from .model import LSTMDrop


def train(
    lat, lon, train_years, seq_length, batch_size, model_name, start_date,
    end_date, hidden_size, num_layers, dropout, epochs, lr, decay, step_size,
    gamma
):
    _, train_loader = dataloader(
        lat, lon, train_years, seq_length, batch_size, model_name,
        norm_save_scaler, start_date, end_date
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMDrop(hidden_size, num_layers, dropout, device).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    print('Start training ...')
    model.train()
    for epoch in range(epochs):
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
        print(
            f'Epoch Summary --> Epoch: {epoch + 1}, '
            f'Epoch Loss: {training_loss / len(train_loader)}, '
            f'LR: {optimizer.param_groups[0]["lr"]}'
        )
    print('Training complete!')
    torch.save(
        model.state_dict(), (f'./objects/trained_models/{model_name}.pt')
    )
