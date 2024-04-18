import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import h5py
import torch.nn.functional as F

import parser
import utils
import model
import load_metrla


args = parser.parse_opt()


(train_x_gp_fw, train_x_gp_bw, train_gp_fw, train_gp_bw, train_TE, train_y,
 val_x_gp_fw, val_x_gp_bw, val_gp_fw, val_gp_bw, val_TE, val_y,
 test_x_gp_fw, test_x_gp_bw, test_gp_fw, test_gp_bw, test_TE, test_y, mean, std) \
 = load_metrla.load_data(args)

start_idx = torch.randint(low = 0, high=args.h, size=(1,)).item()
num_train = (train_x_gp_fw.shape[1] - start_idx) // args.h

end_idx = start_idx + num_train * args.h
train_x_gp_fw_epoch = train_x_gp_fw[:, start_idx : end_idx]
train_x_gp_bw_epoch = train_x_gp_bw[:, start_idx : end_idx]
train_x_gp_fw_epoch = torch.reshape(
    train_x_gp_fw_epoch, shape=(-1, num_train, args.h, args.K, 1)
)
train_x_gp_bw_epoch = torch.reshape(
    train_x_gp_bw_epoch, shape=(-1, num_train, args.h, args.K, 1)
)


model = model.Model(mean, std, args)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)
# criterion = torch.nn.MSELoss()


for epoch in range(args.epochs):
    if args.wait >= args.patience:
        print(f'Early stopping activated in epoch: {epoch}')
        break

    train_loss = []
    start_idx = torch.randint(low = 0, high=args.h, size=(1,)).item()
    num_train = (train_x_gp_fw.shape[1] - start_idx) // args.h
    end_idx = start_idx + num_train * args.h
    train_x_gp_fw_epoch = train_x_gp_fw[:, start_idx : end_idx]
    train_x_gp_bw_epoch = train_x_gp_bw[:, start_idx : end_idx]
    train_x_gp_fw_epoch = torch.reshape(
        train_x_gp_fw_epoch, shape=(-1, num_train, args.h, args.K, 1)
    )
    train_x_gp_bw_epoch = torch.reshape(
        train_x_gp_bw_epoch, shape=(-1, num_train, args.h, args.K, 1)
    )
    train_TE_epoch = train_TE[:, start_idx: end_idx]
    train_TE_epoch = torch.reshape(
        train_TE_epoch, shape=(1, num_train, args.h)
    )
    train_y_epoch = train_y[:, start_idx: end_idx]
    train_y_epoch = torch.reshape(
        train_y_epoch, shape=(-1, num_train, args.h, 1)

    )
    permutation = torch.randperm(num_train)
    train_x_gp_fw_epoch = train_x_gp_fw_epoch[:, permutation]
    train_x_gp_bw_epoch = train_x_gp_bw_epoch[:, permutation]
    train_TE_epoch = train_TE_epoch[:, permutation]
    train_y_epoch = train_y_epoch[:, permutation]

    model.train()
    for i in range(num_train):
        optimizer.zero_grad()
        pred = model(train_x_gp_fw_epoch[:, i], train_x_gp_bw_epoch[:, i],
                     train_TE_epoch[:, i],
                     train_gp_fw, train_gp_bw)
        label = train_y_epoch[:, i]
        loss = utils.mse_loss(pred, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    epoch_loss_train = torch.tensor(train_loss).mean()


    num_val = test_x_gp_fw.shape[1] // args.h
    pred_test = []
    label_test = []
    model.eval()
    with torch.no_grad():
        for i in range(num_val):
            pred = model(test_x_gp_fw[:, i * args.h: (i + 1) * args.h], test_x_gp_bw[:, i*args.h: (i+1)*args.h],
                        test_TE[:, i*args.h: (i+1)*args.h],
                        test_gp_fw, test_gp_bw)
            label = test_y[:, i*args.h: (i+1)*args.h]
            pred_test.append(pred)
            label_test.append(label)

    pred_test = torch.concat(pred_test, dim=1)
    label_test = torch.concat(label_test, dim=1)
    test_rmse, test_mae = utils.metric(pred_test, label_test)
    print(f'Epoch: {epoch} | loss train: {epoch_loss_train} | RMSE: {test_rmse} | MAE: {test_mae}')