import numpy as np
import pandas as pd
import torch
from utils import generate_sample
from torch.utils.data import Dataset, DataLoader
from model import Model

class Dataset_custom(Dataset):
    def __init__(self, file_path, start_idx, train_idx, vali_idx, test_idx, flag='train'):
        assert flag in ['train', 'vali', 'test']
        type_map = {'train':0, 'vali':1, 'test':2}
        self.set_type = type_map[flag]

        self.file_path = file_path
        self.start_index = start_idx
        self.train_index = train_idx
        self.vali_index = vali_idx
        self.test_index = test_idx

        self.__read_data__()

    def __read_data__(self):
        border1s = [self.start_index, self.vali_index, self.train_index]
        border2s = [self.train_index, self.train_index, self.test_index]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df = pd.read_csv(self.file_path)
        total_dates = []
        total_values = []
        for index in range(border1, border2):
            dates, values = generate_sample(df, index)
            total_dates.append(dates)
            total_values.append(values)
        self.total_dates = np.array(total_dates)
        self.total_values = np.array(total_values)

    def __len__(self):
        self.time_period, self.samples = self.total_values.shape[0],self.total_values.shape[1]
        return self.time_period*self.samples*46

    def __getitem__(self, index):
        x = index // (self.samples * 46)
        x_rest = index % (self.samples * 46)
        y = x_rest // (self.samples)
        y_rest = y % (self.samples)
        z = y_rest % 46
        date = self.total_dates[x][y]
        value = self.total_values[x][y][z]
        batch_x = torch.tensor(value[:-1])
        batch_y = torch.tensor(value[-1])
        batch_y = batch_y.unsqueeze(0)
        batch_x_mark = torch.tensor(date[:-1])
        batch_y_mark = torch.tensor(date[-1])
        batch_y_mark = batch_y_mark.unsqueeze(0)
        return batch_x,batch_x_mark,batch_y,batch_y_mark


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    data_set = Dataset_custom(
        file_path=args.file_path,
        start_idx=args.start_idx,
        train_idx=args.train_idx,
        vali_idx=args.vali_idx,
        test_idx=args.test_idx,
        flag=flag
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
