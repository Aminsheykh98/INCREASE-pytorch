import torch

def mse_loss(pred, label):
    mask = torch.not_equal(label, 0.0).type(pred.dtype).requires_grad_(True)
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), 0.0, mask)
    loss = (pred - label) ** 2
    loss *= mask
    loss = torch.where(torch.isnan(loss), 0.0, loss)
    loss = torch.mean(loss)
    return loss

def metric(pred, truth):
    idx = truth > 0
    mask = torch.not_equal(truth, 0.0).type(pred.dtype).requires_grad_(False)
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), 0.0, mask)
    MSE = torch.mean(((pred - truth) ** 2) * mask)
    RMSE = torch.sqrt(MSE)
    MAE = torch.mean(torch.abs(pred - truth) * mask)
    return RMSE, MAE
