## download.py
`download.py` download a year's worth of data from NSRDB

**Flags**
* `--lat`: Latitude (to avoid errors make sure this value is within the continental United States) [Required]
* `--lon`: Longitude (to avoid errors make sure this value is within the continental United States) [Required]
* `--train-years`: Comma separated value string with years to download training data from (1998-2017 according to the official NSRDB docs) [Required]
* `--test-years`: 'Comma separated value string with years to download testing data from (1998-2017 according to the official NSRDB docs) [Required]
* `--interval`: 30 or 60 minute interval data [default: 30]

## train.py
`train.py` train a configurable RNN

**Flags**
* `--lat`: Latitude [Required]
* `--lon`: Longitude [Required]
* `--train-years`: Comma separated value string of downloaded irradiance data [Required]
* `--seq-length`: How many data points are needed to make one prediction [default: 64]
* `--batch-size`: Batch size of the training data [default: 64]
* `--model-name`: Name of the saved model [default: model]
* `--start-date`: Start date if you want to slice [default: None]
* `--end-date`: End date if you want to slice [default: None]
* `--hidden-size`: How many hidden neurons per LSTM layer [default: 35]
* `--num-layers`: How many LSTM layers [default: 2]
* `--dropout`: Dropout rate [default: 0.3]
* `--epochs`: Number of epochs [default: 5]
* `--lr`: Beginning learning rate [default: 1e-2]
* `--decay`: Weight decay also known as L2 penalty [default: 1e-5]
* `--step-size`: Decays the learning rate of each parameter group by gamma every step_size epochs [default: 2]
* `--gamma`: Multiplicative factor of learning rate decay [default: 0.5]

## evaluate.py
`evaluate.py` evaluate and plot the irradiance forecast results of a trained model

**Flags**
* `--lat`: Latitude [Required]
* `--lon`: Longitude [Required]
* `--test-years`: Comma separated value string of downloaded irradiance data [Required]
* `--seq-length`: How many data points are needed to make one prediction [default: 64]
* `--model-name`: Name of the saved model [default: model]
* `--start-date`: Start date if you want to slice [default: None]
* `--end-date`: End date if you want to slice [default: None]
* `--hidden-size`: How many hidden neurons per LSTM layer [default: 35]
* `--num-layers`: How many LSTM layers [default: 2]
* `--plot`: Should we plot the data [default: False]
