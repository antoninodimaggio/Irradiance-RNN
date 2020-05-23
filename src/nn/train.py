import torch
from generate_dataloaders import generate_trainloader
from models import LSTMDrop


def train(train_data_path, test_data_path, model_name, batch_size=64, seq_length=64,
          hidden_size=35, num_layers=2, dropout=0.3, num_epochs=8):
    device = torch.device('cuda')
    train_loader = generate_trainloader(train_data_path, test_data_path,
                                        seq_length, batch_size)
    input_size = 1
    model = LSTMDrop(input_size, hidden_size, num_layers, dropout, device).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    print('Start training ...')
    for epoch in range(num_epochs):
        training_loss = 0.0
        model.train()
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
    print('Training complete!\n')
    torch.save(model.state_dict(), ('./trained_models/' + model_name + '.pt'))
