import argparse
from rnn.train_evaluate_plot import train

def main():
    # long lines
    parser = argparse.ArgumentParser(description='Train a configurable RNN')
    parser.add_argument('--train-data-path', type=str, help='Path to training data [Required]', required=True)
    parser.add_argument('--test-data-path', type=str, help='Path to testing data [Required]', required=True)
    parser.add_argument('--name', type=str, default='model', help='Name of the saved model (default: model)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size of the training data (default: 64)')
    parser.add_argument('--seq-length', type=int, default=64, help='How many data points are needed to make one prediction (default: 64)')
    parser.add_argument('--hidden-size', type=int, default=35, help='How many hidden neurons per LSTM layer (default: 35)')
    parser.add_argument('--num-layers', type=int, default=2, help='How many LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-2, help='Beginning learning rate (default: 1e-2)')
    parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay also known as L2 penalty (default: 1e-5)')
    parser.add_argument('--step-size', type=int, default=2, help='Decays the learning rate of each parameter group by gamma every step_size epochs (default: 2)')
    parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay (default: 0.5)')
    args = parser.parse_args()
    train(args.train_data_path, args.test_data_path, args.name, args.batch_size,
          args.seq_length, args.hidden_size, args.num_layers, args.dropout,
          args.epochs, args.lr, args.decay, args.step_size, args.gamma)

if __name__ == '__main__':
    main()
