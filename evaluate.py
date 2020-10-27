import argparse

from irradiance_rnn.evaluate import evaluate
from irradiance_rnn.plot import pretty_plot


def main():
    # long lines
    parser = argparse.ArgumentParser(
        description=
        'Evaluate and plot the irradiance forecast results of a trained model'
    )
    parser.add_argument(
        '--lat', type=float, required=True, help='Latitude [Required]'
    )
    parser.add_argument(
        '--lon', type=float, required=True, help='Longitude [Required]'
    )
    parser.add_argument(
        '--test-years',
        type=str,
        required=True,
        help=
        'Comma seperated value string of downloaded irradaince data [Required]',
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=64,
        help=
        'How many data points are needed to make one prediction [default: 64]'
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
        '--plot',
        action='store_true',
        default=False,
        help='Should we plot the data [default: False]'
    )
    args = vars(parser.parse_args())
    dates, predicted, actual, rmse = evaluate(**args)
    if args['plot']:
        pretty_plot(dates, predicted, actual, round(rmse, 2))


if __name__ == '__main__':
    main()
