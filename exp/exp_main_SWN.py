from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.SpectralWaveletNet import SpectralWaveletNet
from utils.tools import EarlyStopping, adjust_learning_rate ,visual, test_params_flop
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
warnings.filterwarnings('ignore')

class Exp_Main_SWN(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_SWN, self).__init__(args)
        self.min_test_loss = np.inf
        self.epoch_for_min_test_loss = 0
        
    def _build_model(self):
        # Initialize SpectralWaveletNet (Clean Version)
        model = SpectralWaveletNet(
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            c_in=self.args.c_in,
            d_model=self.args.d_model,
            dropout=self.args.dropout,
            wave_level=self.args.level,   
            wave_basis=self.args.wavelet, 
            device=self.device
        ).float()
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_criterion(self): # 选择损失函数
        criterion = {'mse': torch.nn.MSELoss(), 'smoothL1': torch.nn.SmoothL1Loss()}
        return criterion[self.args.loss] 

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()        
        preds_mean, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                # Validation uses standard output
                pred_mean, true = self._process_one_batch(vali_data, batch_x, batch_y, mode='val')
                preds_mean.append(pred_mean)
                trues.append(true)
            preds_mean = torch.cat(preds_mean).cpu()
            trues = torch.cat(trues).cpu()
            
            preds_mean = preds_mean.reshape(-1, preds_mean.shape[-2], preds_mean.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe = metric(preds_mean.numpy(), trues.numpy())
            self.model.train()
            return mse, mae

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion() 

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler(init_scale = 1024)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none = True)
                
                batch_x = batch_x.to(dtype=torch.float, device=self.device)
                batch_y = batch_y.to(dtype=torch.float, device=self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # Retrieve components
                        pred, pred_yl, pred_yh = self.model(batch_x, return_decomposition=True)
                        # Calculate DBLoss
                        loss = self._compute_db_loss(batch_y, pred, pred_yl, pred_yh, criterion)
                else:
                    pred, pred_yl, pred_yh = self.model(batch_x, return_decomposition=True)
                    loss = self._compute_db_loss(batch_y, pred, pred_yl, pred_yh, criterion)
                
                train_loss.append(loss.item())
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch {}: cost time: {:.2f} sec".format(epoch + 1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae = self.vali(test_data, test_loader, criterion)

            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss

            print("\tEpoch {0}: Steps- {1} | Train Loss: {2:.5f} Vali.MSE: {3:.5f} Test.MSE: {4:.5f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("\tEarly stopping")
                break
            adjust_learning_rate(model_optim, None, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def _compute_db_loss(self, true, pred, pred_yl, pred_yh_list, criterion):
        """
        Calculates DBLoss: 
        Decomposes Ground Truth (True) into Trend and Detail, then computes loss per component.
        """
        B, L, C = true.shape
        if self.args.use_multi_gpu:
            revin = self.model.module.global_revin
            dwt = self.model.module.dwt
        else:
            revin = self.model.global_revin
            dwt = self.model.dwt
            
        # 1. Normalize Ground Truth to align with latent space
        true_norm = revin(true, 'norm')
        true_norm = true_norm.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # 2. Decompose Ground Truth
        # (Note: This adds computational overhead but ensures DBLoss correctness)
        true_yl, true_yh_list = dwt(true_norm)
        
        # 3. Component Supervision
        # Trend Loss
        loss_trend = criterion(pred_yl, true_yl)
        
        # Detail Loss
        loss_detail = 0
        for p_h, t_h in zip(pred_yh_list, true_yh_list):
            loss_detail += criterion(p_h, t_h)
            
        # 4. Reconstruction Loss
        loss_recon = criterion(pred, true)
        
        # Weighted Sum (Explicit Attention to components)
        total_loss = loss_recon + loss_trend + loss_detail
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch_x, batch_y, mode='test')
                preds.append(pred)
                trues.append(true)
            preds = torch.cat(preds).cpu()
            trues = torch.cat(trues).cpu()
            
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
            print('mse: {}, mae: {}'.format(mse, mae))
            
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            f = open("result_SWN.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')
            f.close()
            return mse, mae

    def _process_one_batch(self, dataset_object, batch_x, target, mode):
        batch_x = batch_x.to(dtype = torch.float, device = self.device)
        target =  target.to(dtype = torch.float, device = self.device)
        
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                pred = self.model(batch_x, return_decomposition=False)
        else:
            pred = self.model(batch_x, return_decomposition=False)
        return pred, target
