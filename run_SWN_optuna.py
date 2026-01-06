import argparse
import torch
import optuna 
import os
import random
import numpy as np
import time
from math import log2
from exp.exp_main_SWN import Exp_Main_SWN
from utils.tools import set_random_seed, adjust_learning_rate, EarlyStopping

class Exp_Optuna(Exp_Main_SWN):
    def __init__(self, args, trial):
        super(Exp_Optuna, self).__init__(args)
        self.trial = trial

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler(init_scale=1024)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad(set_to_none=True)
                pred_mean, true = self._process_one_batch(train_data, batch_x, batch_y, 'train')
                loss = criterion(pred_mean, true)
                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion)
            
            # --- PRUNING ---
            self.trial.report(vali_loss, epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            # ---------------

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

def objective(trial):
    opt_args = argparse.Namespace(**vars(base_args))
    



    opt_args.seq_len = trial.suggest_categorical("seq_len", [96, 192, 336, 512])

    opt_args.wavelet = trial.suggest_categorical("wavelet", 
        [
            "db4", "db6", "sym3","sym4","sym8","coif2","coif5"
        ])
    
    max_safe_level = int(log2(opt_args.seq_len)) - 3 # Stricter safety margin for long filters
    search_level_high = min(4, max(1, max_safe_level))
    opt_args.level = trial.suggest_int("level", 2, search_level_high)

    opt_args.d_model = trial.suggest_categorical("d_model", [64, 96, 128, 256])
    
    opt_args.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    opt_args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    opt_args.learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
    opt_args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # ==========================================
    
    setting = '{}_OPT_V4_{}_sl{}_wv{}_lv{}_dm{}_dr{}_wd{}_bs{}'.format(
        opt_args.model, opt_args.data, 
        opt_args.seq_len, opt_args.wavelet, opt_args.level,
        opt_args.d_model, int(opt_args.dropout*100), opt_args.weight_decay, opt_args.batch_size)
    
    try:
        exp = Exp_Optuna(opt_args, trial)
        exp.train(setting)
        vali_mse, vali_mae = exp.test(setting)
        return vali_mse
        
    except RuntimeError as e:
        err_msg = str(e)
        if "out of memory" in err_msg:
            torch.cuda.empty_cache()
            print(f"Trial pruned due to OOM.")
            raise optuna.exceptions.TrialPruned()
        elif "filters" in err_msg or "Wavelet" in err_msg or "size" in err_msg:
            # This catches invalid wavelet/length combinations (e.g. signal too short for db16)
            print(f"Trial pruned due to Wavelet/Length incompatibility: {opt_args.wavelet} on len {opt_args.seq_len}")
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[Optuna V4] SWN Comprehensive Wavelet Tuning')

    # Basic Config
    parser.add_argument('--n_trials', type=int, default=60)
    parser.add_argument('--model', type=str, default='SpectralWaveletNet')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Fixed Args
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=0)
    
    # Defaults (Overridden)
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--wavelet', type=str, default='db4')
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Training
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=20) 
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--use_amp', action='store_true', default=False)
    
    # Data Loader
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')
    parser.add_argument('--seed', type=int, default=42)

    base_args = parser.parse_args()
    
    base_args.use_gpu = True if torch.cuda.is_available() and base_args.use_gpu else False
    if base_args.use_gpu and base_args.use_multi_gpu:
        base_args.devices = base_args.devices.replace(' ','')
        device_ids = base_args.devices.split(',')
        base_args.device_ids = [int(id_) for id_ in device_ids]
        base_args.gpu = base_args.device_ids[0]

    # Initialize Dataset
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'T': 'OT', 'M': [21, 21], 'S': [1, 1], 'MS': [21, 1]},
        'Traffic': {'data': 'traffic.csv', 'root_path': './data/traffic/', 'T': 'OT', 'M': [862, 862], 'S': [1, 1], 'MS': [862, 1]},
        'Electricity': {'data': 'electricity.csv', 'root_path': './data/electricity/', 'T': 'OT', 'M': [321, 321], 'S': [1, 1], 'MS': [321, 1]},
        'ILI':  {'data': 'national_illness.csv', 'root_path': './data/illness/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Solar':  {'data': 'solar_AL.txt', 'root_path': './data/solar/', 'T': None, 'M': [137, 137], 'S': [None, None], 'MS': [None, None]},
    }
    if base_args.data in data_parser.keys():
        data_info = data_parser[base_args.data]
        base_args.data_path = data_info['data']
        base_args.root_path = data_info['root_path']
        base_args.target = data_info['T']
        base_args.c_in = data_info[base_args.features][0]

    print("Starting V4 Comprehensive Wavelet Tuning...")
    
    # TPE Sampler for smart search over the large categorical space
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=base_args.n_trials)

    print("\n" + "="*20 + " RESULTS V4 " + "="*20)
    print("Best Trial:")
    trial = study.best_trial
    print(f"  MSE: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
