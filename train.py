import argparse

from irradiance_rnn.train import train


def main():
    parser = argparse.ArgumentParser(description='Train a configurable RNN')
    parser.add_argument(
        '--lat', type=float, required=True, help='Latitude [Required]'
    )
    parser.add_argument(
        '--lon', type=float, required=True, help='Longitude [Required]'
    )
    parser.add_argument(
        '--train-years',
        type=str,
        required=True,
        help=
        'Comma seperated value string of downloaded irradaince data [Required]'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=64,
        help=
        'How many data points are needed to make one prediction [default: 64]'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size of the training data [default: 64]'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='model',
        help='Name of the saved model [default: model]'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date if you want to slice [default: None]'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date if you want to slice [default: None]'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=35,
        help='How many hidden neurons per LSTM layer [default: 35]'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='How many LSTM layers [default: 2]'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate [default: 0.3]'
    )
    parser.add_argument(
        '--epochs', type=int, default=5, help='Number of epochs [default: 5]'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-2,
        help='Beginning learning rate [default: 1e-2]'
    )
    parser.add_argument(
        '--decay',
        type=float,
        default=1e-5,
        help='Weight decay also known as L2 penalty [default: 1e-5]'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=2,
        help=
        'Decays the learning rate of each parameter group by gamma every step_size epochs [default: 2]'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='Multiplicative factor of learning rate decay [default: 0.5]'
    )
    args = vars(parser.parse_args())
    train(**args)


if __name__ == '__main__':
    main()
