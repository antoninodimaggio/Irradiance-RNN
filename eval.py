import argparse
from rnn.train_evaluate_plot import pretty_plot, test

def main():
    # long lines
    parser = argparse.ArgumentParser(description='Evaluate and plot the irradiance forecast results of a trained model')
    parser.add_argument('--test-data-path', type=str, help='Path to testing data [Required]', required=True)
    parser.add_argument('--name', type=str, default='model', help='Name of the saved model (default: model)')
    parser.add_argument('--seq-length', type=int, default=64, help='How many data points are need to make one prediction (default: 64)')
    parser.add_argument('--hidden-size', type=int, default=35, help='How many hidden neurons per LSTM layer (default: 35)')
    parser.add_argument('--num-layers', type=int, default=2, help='How many LSTM layers (default: 2)')
    parser.add_argument('--plot', action='store_true', default=False, help='Should we plot the data (default: False)')
    parser.add_argument('--s-split', type=int, default=0, help='At what data point should we start plotting (default: 0)')
    parser.add_argument('--e-split', type=int, default=800,
        help='At what data point should we stop plotting, if you exceed the size of the dataset then \
        it will default to plotting the whole dataset (default: 800)')
    parser.add_argument('--plot-name', type=str, default='test', help='Name of the saved plot (default: test)')
    args = parser.parse_args()
    dates, predicted, actual = test(args.test_data_path, args.name, args.seq_length,
                                    args.hidden_size, args.num_layers)
    if args.plot:
        pretty_plot(dates, predicted, actual, args.seq_length, args.s_split,
                    args.e_split, args.plot_name)

if __name__ == '__main__':
    main()
