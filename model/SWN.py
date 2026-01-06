import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class SpectralBranch(nn.Module):
    """
    Standard Spectral Branch (Dense MLP)
    Maps past frequency coefficients to future frequency coefficients.
    """
    def __init__(self, in_len, out_len, hidden_dim=256, dropout=0.1):
        super(SpectralBranch, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        
        # Frequency Dimensions (rfft)
        self.freq_in = in_len // 2 + 1
        self.freq_out = out_len // 2 + 1
        
        # MLP Dimensions (Real + Imag parts concatenated)
        self.mlp_in_dim = self.freq_in * 2
        self.mlp_out_dim = self.freq_out * 2
        
        self.revin = RevIN(num_features=1, affine=True)
        
        # Dense MLP (High Capacity)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.mlp_out_dim)
        )

    def forward(self, x):
        # x shape: [Batch*Channel, 1, Length_in]
        B, C, L = x.shape
        
        # 1. Branch Norm
        x = x.transpose(1, 2) 
        x = self.revin(x, 'norm')
        x = x.transpose(1, 2) 
        
        # 2. FFT
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        
        # 3. Flatten Complex -> Real
        x_vals = torch.view_as_real(x_fft)
        x_flat = x_vals.reshape(B, C, -1)
        
        # 4. Dense MLP Forecast
        y_flat = self.mlp(x_flat)
        
        # 5. Reshape Real -> Complex
        y_vals = y_flat.reshape(B, C, self.freq_out, 2)
        y_fft = torch.view_as_complex(y_vals.contiguous())
        
        # 6. IFFT
        y = torch.fft.irfft(y_fft, n=self.out_len, dim=-1, norm='ortho')
        
        # 7. Branch Denorm
        y = y.transpose(1, 2)
        y = self.revin(y, 'denorm')
        y = y.transpose(1, 2)
        
        return y

class SpectralWaveletNet(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, d_model=256, dropout=0.1, 
                 wave_level=3, wave_basis='db4', device=torch.device('cuda')):
        super(SpectralWaveletNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.device = device
        self.wave_level = wave_level
        self.wave_basis = wave_basis
        
        self.global_revin = RevIN(c_in)
        
        # Hardware Accelerated Wavelet
        self.dwt = DWT1DForward(wave=wave_basis, J=wave_level).to(device)
        self.idwt = DWT1DInverse(wave=wave_basis).to(device)
        
        # Dynamic Shape Calculation
        self.in_shapes, self.out_shapes = self._calculate_shapes()
        
        # Independent Branches
        self.branches = nn.ModuleList()
        
        # Approx Branch (Trend)
        self.branches.append(
            SpectralBranch(self.in_shapes[0], self.out_shapes[0], hidden_dim=d_model, dropout=dropout)
        )
        
        # Detail Branches (Seasonality/Noise)
        for i in range(len(self.in_shapes) - 1):
            self.branches.append(
                SpectralBranch(self.in_shapes[i+1], self.out_shapes[i+1], hidden_dim=d_model, dropout=dropout)
            )
            
    def _calculate_shapes(self):
        dummy_in = torch.randn(1, 1, self.seq_len).to(self.device)
        yl, yh = self.dwt(dummy_in)
        in_shapes = [yl.shape[-1]] + [d.shape[-1] for d in yh]
        
        dummy_out = torch.randn(1, 1, self.pred_len).to(self.device)
        yl_out, yh_out = self.dwt(dummy_out)
        out_shapes = [yl_out.shape[-1]] + [d.shape[-1] for d in yh_out]
        return in_shapes, out_shapes

    def forward(self, x, return_decomposition=False):
        # x: [Batch, Seq_Len, Channels]
        B, L, C = x.shape
        
        x = self.global_revin(x, 'norm')
        # Channel Independence: Treat channels as batch
        x = x.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # Decompose Input
        yl_in, yh_in = self.dwt(x)
        
        # Branch Processing
        yl_out = self.branches[0](yl_in)
        yh_out = []
        for i, detail_coeff in enumerate(yh_in):
            pred_detail = self.branches[i+1](detail_coeff)
            yh_out.append(pred_detail)
            
        # Reconstruct
        y_pred = self.idwt((yl_out, yh_out))
        
        # Reshape back
        y_pred = y_pred.reshape(B, C, -1).permute(0, 2, 1)
        y_pred = self.global_revin(y_pred, 'denorm')
        
        if return_decomposition:
            # [DBLoss] Return components to supervise branches
            return y_pred, yl_out, yh_out
        
        return y_pred
