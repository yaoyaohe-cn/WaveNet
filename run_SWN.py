import argparse
import torch
import random 
import numpy as np
from exp.exp_main_SWN import Exp_Main_SWN
from utils.tools import set_random_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '[SpectralWaveletNet] Long Sequences Forecasting')
    

    parser.add_argument('--model', type=str, required=False, default='SpectralWaveletNet')
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, classification, anomaly_detection]')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    

    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    parser.add_argument('--embed', type=str, default='timeF', 
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h', 
                        help='freq for time features encoding')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', 
                        help='subset for M4 dataset')
    

    parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    

    parser.add_argument('--d_model', type=int, default=256, help='Hidden dimension of the Spectral MLP')
    parser.add_argument('--wavelet', type=str, default='db4', help='wavelet basis (e.g., db2, db4)')
    parser.add_argument('--level', type=int, default=3, help='wavelet decomposition level')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    

    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='optimizer weight decay')
    parser.add_argument('--loss', type=str, default='smoothL1', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
    

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'T': 'OT', 'M': [21, 21], 'S': [1, 1], 'MS': [21, 1]},
        'Traffic': {'data': 'traffic.csv', 'root_path': './data/traffic/', 'T': 'OT', 'M': [862, 862], 'S': [1, 1], 'MS': [862, 1]},
        'Electricity': {'data': 'electricity.csv', 'root_path': './data/electricity/', 'T': 'OT', 'M': [321, 321], 'S': [1, 1], 'MS': [321, 1]},
    }
    
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.root_path = data_info['root_path']
        args.target = data_info['T']
        args.c_in = data_info[args.features][0]

    print('Args in experiment: {}'.format(args))
    
    setting = '{}_{}_sl{}_pl{}_bs{}dm{}_wv{}_lv{}_dr{}_wd{}_sd{}'.format(
        args.model, args.data, 
        args.seq_len, args.pred_len, 
        args.batch_size, args.d_model, 
        args.wavelet, args.level,
        args.dropout, args.weight_decay,
        args.seed)
    
    set_random_seed(args.seed)
    
    exp = Exp_Main_SWN(args)
    
    print(f'>>>>>>> Start Training : {setting} >>>>>>>>>')
    exp.train(setting)
    
    print(f'>>>>>>> Start Testing : {setting} >>>>>>>>>')
    exp.test(setting)
