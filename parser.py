import argparse
import torch


def parse_opt():

    # Settings
    parser = argparse.ArgumentParser()
    path_h5 = './dataset/metr-la/metr-la.h5'
    path_ids = './dataset/metr-la/graph_sensor_ids.txt'
    path_dist = './dataset/metr-la/distances_la_2012.csv'
    path_test_nodes = './dataset/metr-la/test_node_increase.npy'

    # Data loading args
    parser.add_argument('--percent_train_samples', type=float, default='0.6', help='Training time window percentage')
    parser.add_argument('--percent_val_samples', type=float, default='0.2', help='Validationg time window percentage')
    parser.add_argument('--node_percent_test', type=float, default='0.2', help='Percentage of test nodes')
    parser.add_argument('--node_percent_val', type=float, default='0.2', help='Percentage of validation nodes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--path_h5', type=str, default=path_h5, help='Metr-LA data file path which contains sensor readings')
    parser.add_argument('--path_ids', type=str, default=path_ids, help='Metr-LA sensor ids path which contains sensor indices')
    parser.add_argument('--path_dist', type=str, default=path_dist, help='Metr-LA pre-defined distances path')
    parser.add_argument('--path_test_nodes', type=str, default=path_test_nodes, help='Test nodes indices file')
    parser.add_argument('--normalized_k', type=float, default='0.1', help='Normalization factor to impose sparsity for graph adjacency matrix')
    parser.add_argument('--K', type=int, default='15', help='Number of neighbor nodes')
    parser.add_argument('--device', type=str, default='auto', help='Device')


    # Model args
    parser.add_argument('--h', type=int, default='24', help='length time window')
    parser.add_argument('--wait', type=int, default='0', help='')
    parser.add_argument('--d', type=int, default='64', help='hidden layer dimension')
    


    # Train args
    parser.add_argument('--epochs', type=int, default='20', help='Number of epochs')
    parser.add_argument('--patience', type=int, default='5', help='Number of epochs before activating early stopping')
    parser.add_argument('--N_target', type=int, default='104', help='Number of targets')
    parser.add_argument('--lr', type=float, default='0.001', help='Learning rate')



    args, unknowns = parser.parse_known_args()


    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    return args