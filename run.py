import argparse
import torch
import random
import numpy as np
from exp import Exp_prediction

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='Transformer')
    # data loader
    parser.add_argument('--file_path', type=str, default='Profile_10.csv')
    parser.add_argument('--start_idx', type=int, default=20, help='training dataset start index')
    parser.add_argument('--train_idx', type=int, default=25, help='validation dataset start index')
    parser.add_argument('--vali_idx', type=int, default=23, help='testing dataset start index')
    parser.add_argument('--test_idx', type=int, default=27, help='end of dateset index')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # model define
    parser.add_argument('--emb_dim', type=int, default=128, help='number of embeded dimentions')
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='number of decoder layers')
    parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')

    #GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    #optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=128, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in exeriment:\n', args)
    for ii in range(args.itr):
        setting = 'ed{}_nh{}_el{}_dl{}_{}'.format(
                args.emb_dim,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                ii)
    exp = Exp_prediction(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    torch.cuda.empty_cache()

