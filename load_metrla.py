import numpy as np
import torch
import pandas as pd
import h5py
import functools


def heterogeneous_relations(A, train_node, K):
    relation = np.zeros_like(A) - 1
    relation[:, train_node] = A[:, train_node] # only use the train nodes
    np.fill_diagonal(relation, val = -1) # delete self-loop connections
    Neighbor = np.argsort(-relation, axis = 1) # descending order
    Neighbor = Neighbor[:, : K]
    relation = -np.sort(-relation, axis = -1)
    relation = relation[:, : K]
    relation[relation < 0] = 0
    relation = relation / (1e-10 + np.sum(relation, axis = 1, keepdims = True))
    return relation, Neighbor


def _to_torch(device, input_array):
    return torch.tensor(input_array, device=device)

def load_data(args):
    rand = np.random.RandomState(args.seed)
    # x
    df = pd.DataFrame(np.array(h5py.File(args.path_h5)['df']['block0_values']))
    distance_df = pd.read_csv(args.path_dist, dtype={'from': 'str', 'to': 'str'})
    nodes = list(df.columns)
    x = df.values.astype(np.float32).T
    N, num_interval = x.shape
    # TE
    minutes_1h = np.arange(0, 60, 5)
    minutes_1d = np.tile(minutes_1h, 24)
    minutes_total = np.tile(minutes_1d, reps= 34272//288)
    hours_1d = np.array(np.arange(0, 24, 1/12), dtype=int)
    hours_total = np.tile(hours_1d, reps= 34272//288)

    TE = (hours_total * 3600 + minutes_total * 60) // (24 * 3600 / 288)

    TE = np.array(TE).astype(np.int32)
    TE = TE[np.newaxis]
    # node
    # test_node = rand.choice(list(range(0, N)),int(0.25 * N),replace=False)
    test_node = np.load(args.path_test_nodes)
    print(test_node)
##    test_node = np.load(args.test_file)
    train_node = np.setdiff1d(np.arange(N), test_node)
    np.random.shuffle(train_node)
    # Geo-proximity
    # dist = np.loadtxt(path_dist, delimiter = ',', skiprows = 1)
    dist_mx = np.zeros(shape = (N, N), dtype = np.float32)
    dist_mx[:] = np.inf

    with open(args.path_ids) as f:
        sensor_ids = f.read().strip().split(',')

    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    dist_train = dist_mx[train_node, :][:, train_node]
    std = np.std(dist_train[~np.isinf(dist_train)])

    A_gp = np.exp(-dist_mx ** 2 / std ** 2)
    gp_fw, Neighbor_gp_fw = heterogeneous_relations(A_gp, train_node, args.K)
    gp_bw, Neighbor_gp_bw = heterogeneous_relations(A_gp.T, train_node, args.K)
    # train/val/test
    num_train = int(0.7 * num_interval)
    num_val = int(0.2 * num_train)

    train_data = x[train_node, :num_train]
    std, mean = np.std(train_data), np.mean(train_data)


    num_test = num_interval - num_train #- num_val
    train_TE = TE[:, : num_train]
    val_TE = TE[:, num_train : num_train + num_val]
    test_TE = TE[:, -num_test :]
    train_x_gp_fw = np.transpose(
        x[Neighbor_gp_fw[train_node], : num_train, np.newaxis],
        axes = (0, 2, 1, 3))
    train_x_gp_bw = np.transpose(
        x[Neighbor_gp_bw[train_node], : num_train, np.newaxis],
        axes = (0, 2, 1, 3))
    train_y = x[train_node, : num_train, np.newaxis]
    val_x_gp_fw = np.transpose(
        x[Neighbor_gp_fw[train_node], num_train : num_train + num_val, np.newaxis],
        axes = (0, 2, 1, 3))
    val_x_gp_bw = np.transpose(
        x[Neighbor_gp_bw[train_node], num_train : num_train + num_val, np.newaxis],
        axes = (0, 2, 1, 3))
    val_y = x[train_node, num_train : num_train + num_val, np.newaxis]
    test_x_gp_fw = np.transpose(
        x[Neighbor_gp_fw[test_node], -num_test :, np.newaxis],
        axes = (0, 2, 1, 3))
    test_x_gp_bw = np.transpose(
        x[Neighbor_gp_bw[test_node], -num_test :, np.newaxis],
        axes = (0, 2, 1, 3))
    test_y = x[test_node, -num_test :, np.newaxis]
    train_gp_fw = gp_fw[train_node, np.newaxis, np.newaxis]
    train_gp_bw = gp_bw[train_node, np.newaxis, np.newaxis]
    val_gp_fw = gp_fw[train_node, np.newaxis, np.newaxis]
    val_gp_bw = gp_bw[train_node, np.newaxis, np.newaxis]
    test_gp_fw = gp_fw[test_node, np.newaxis, np.newaxis]
    test_gp_bw = gp_bw[test_node, np.newaxis, np.newaxis]

    to_torch = functools.partial(_to_torch, args.device)

    
    return (to_torch(train_x_gp_fw), to_torch(train_x_gp_bw), to_torch(train_gp_fw),
            to_torch(train_gp_bw), to_torch(train_TE), to_torch(train_y),
            to_torch(val_x_gp_fw), to_torch(val_x_gp_bw), to_torch(val_gp_fw),
            to_torch(val_gp_bw), to_torch(val_TE), to_torch(val_y),
            to_torch(test_x_gp_fw), to_torch(test_x_gp_bw), to_torch(test_gp_fw),
            to_torch(test_gp_bw), to_torch(test_TE), to_torch(test_y),
            to_torch(mean), to_torch(std))