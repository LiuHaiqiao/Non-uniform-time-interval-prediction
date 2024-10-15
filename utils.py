import torch
from datetime import datetime
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            10: 1e-3, 20: 5e-4, 40: 1e-5, 60: 5e-6,
            80: 1e-7, 100: 1e-8, 120: 1e-9, 160:1e-10
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

# date_loader tool
def date_to_tensor(dates):
    tensor_data = []
    for date in dates:
        # Parse the date (assuming format 'YYYY/MM/DD')
        dt = datetime.strptime(date, '%Y/%m/%d')
        # Normalize year (e.g., map 2020 to 0, 2021 to 1, etc.)
        year = dt.year - 2020  # Modify based on the range of years you're using
        month = dt.month
        day = dt.day
        weekday = dt.weekday()  # Monday is 0, Sunday is 6
        # Append the [year, month, day, weekday] to the list
        tensor_data.append([year, month, day, weekday])
    # Convert list to a tensor
    return tensor_data


def sample_sliding_window(input, window_size=200, step_size=100):
    num_samples = (input.shape[1] - window_size) // step_size + 1
    sampled_windows = []
    for i in range(num_samples):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = input[:, start_idx:end_idx]
        sampled_windows.append(window)
    return np.array(sampled_windows)
def generate_sample(data, index, samples=100):
    full_range = list(range(index - 30, index))
    combinations = [sorted(random.sample(full_range, 20) + [index]) for _ in range(samples)]
    dates = []
    values = []
    for comb in combinations:
        select_columns = data.iloc[:, comb]
        date = select_columns.columns
        date = date_to_tensor(date)
        dates.append(date)
        value = select_columns.values
        value = sample_sliding_window(value.T)
        values.append(value)

    return dates, values

def visual(true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


