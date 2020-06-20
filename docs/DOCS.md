## download.py
`download.py` download a year's worth of data from NSRDB

**Flags**
* `--lat`: Latitude (to avoid errors make sure this value is within the continental United States) [Required]
* `--lon`: Longitude (to avoid errors make sure this value is within the continental United States) [Required]
* `--year`: Year (1998-2017 according to the official NSRDB docs) [Required]
* `--leap-year`: Is it a leap year, make sure the string is either true or false (default: false)
* `--interval`: 30 or 60 minute interval data (default: 30)

## train.py
`train.py` train a configurable RNN

**Flags**
* `--train-data-path`: Path to training data [Required]
* `--test-data-path`: Path to testing data [Required]
* `--name`: Name of the saved model (default: model)
* `--batch-size`: Batch size of the training data (default: 64)
* `--seq-length`: How many data points are needed to make one prediction (default: 64)
* `--hidden-size`: How many hidden neurons per LSTM layer (default: 35)
* `--num-layers`: How many LSTM layers (default: 2)
* `--dropout`: Dropout rate (default: 0.3)
* `--epochs`: Number of epochs (default: 8)
* `--lr`: Beginning learning rate (default: 1e-2)
* `--decay`: Weight decay also known as L2 penalty (default: 1e-5)
* `--step-size`: Decays the learning rate of each parameter group by gamma every step_size epochs (default: 2)
* `--gamma`: Multiplicative factor of learning rate decay (default: 0.5)

## eval.py
`eval.py` evaluate and plot the irradiance forecast results of a trained model

**Flags**
* `--test-data-path`: Path to testing data [Required]
* `--name`: Name of the saved model (default: model)
* `--seq-length`: How many data points are need to make one prediction (default: 64)
* `--hidden-size`: How many hidden neurons per LSTM layer (default: 35)
* `--num-layers`: How many LSTM layers (default: 2)
* `--plot`: Should we plot the data (default: False)
* `--s-split`: At what data point should we start plotting (default: 0)
* `--e-split`: At what data point should we stop plotting, if you exceed the size of the dataset then it will default to plotting the whole dataset (default: 800)
* `--plot-name`: Name of the saved plot (default: test)
