from test import test, pretty_plot
from train import train


############################# THINGS YOU CAN EDIT ##############################
# should we train a new model
should_train = True
# path to a years worth of training data
train_data = '../../data/33.2164_-97.1292_2010.csv'
# path to a years worth of test data
test_data = '../../data/33.2164_-97.1292_2011.csv'
# name of the model ./trained_models/model_name.pt (exclude .pt in model_name)
model_name = 'texas'
# bacth size of training data
batch_size=64
# sequence length when training
seq_length=64
# number of hidden neurons per LSTM
hidden_size=35
# number of LSTM layers
num_layers=2
# dropout rate
dropout=0.3
# number of epochs to train for
num_epochs=8
# should we test and plot (if so then worry about the rest of the params)
should_test = True
# what row to start at when testing
start = 0
# what row to end at when testing ~8,000 usally for 30 minute data
end = 300
# title of the plot
plot_title = 'Predicted vs Actual (Denton County, Texas)'
# name of the saved image ../../images/image_name.png (exclude .png in image_name)
image_name = 'denton_county_texas'
################################################################################
# NOTE: MAKE SURE EACH ARGUMENT IS CORRECTLY INSTANTIATED NO MATTER THE MODE
if (should_train == True):
    train(train_data, test_data, model_name,
          batch_size=batch_size, seq_length=seq_length, hidden_size=hidden_size,
          num_layers=num_layers, dropout=dropout, num_epochs=num_epochs)

if (should_test == True):
    dates, predicted, actual = test(train_data, test_data, model_name, start, end,
                                    seq_length=seq_length, hidden_size=hidden_size,
                                    num_layers=num_layers, dropout=dropout)

    pretty_plot(dates, predicted, actual, image_name, plot_title)
