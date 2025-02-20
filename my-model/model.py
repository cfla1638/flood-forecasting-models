import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

class TemporalFiLM(nn.Module):
    def __init__(self, feature_dim, condition_dim, timesteps):
        """
        Parameters:
         - feature_dim: 输入的特征维度
         - condition_dim: 条件的维度
         - timesteps: 时间步长度
        """
        super(TemporalFiLM, self).__init__()
        self.feature_dim = feature_dim
        self.timesteps = timesteps
        
        # 为固定时间步长生成γ和β参数
        self.fc_gamma = nn.Linear(condition_dim, feature_dim * timesteps)
        self.fc_beta = nn.Linear(condition_dim, feature_dim * timesteps)
        
    def forward(self, x : torch.Tensor, condition : torch.Tensor) -> torch.Tensor:
        """
        Parameters:
         - x: 输入的时间序列数据，[batch, T, feature_dim]
         - condition: 条件数据，[batch, condition_dim]
        Return: FiLM后的数据，[batch, T, feature_dim]
        """
        batch_size, T, _ = x.shape
        
        # 确保输入的时间步长与模块初始化时指定的时间步长一致
        assert T == self.timesteps, f"Expected timesteps {self.timesteps}, but got {T}"
        
        # 生成所有时间步的gamma和beta参数
        gamma = self.fc_gamma(condition)  # [batch, feature_dim * timesteps]
        beta = self.fc_beta(condition)    # [batch, feature_dim * timesteps]
        
        # 重塑形状以匹配时间维度
        # 从 [batch, feature_dim * timesteps] -> [batch, timesteps, feature_dim]
        gamma = gamma.view(batch_size, self.timesteps, self.feature_dim)
        beta = beta.view(batch_size, self.timesteps, self.feature_dim)
        
        # 应用FiLM变换
        return gamma * x + beta

class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_head: int, dropout: float = 0.1) -> None:
        """
        Parameters:
         - dim: 输入的维度
         - num_head: 多头自注意力的头数
         - dropout: dropout的概率
        """
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(dim, num_head, batch_first=True, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, 3, padding='same')     # 一维卷积
        self.layer_norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameter:
         - x: (batch_size, seq_len, input_dim)
        Return: (batch_size, seq_len, input_dim)
        """
        attn_output, _ = self.multi_head_attention(x, x, x) # (batch_size, seq_len, input_dim)
        x = self.layer_norm1(x + self.dropout(attn_output)) # (batch_size, seq_len, input_dim)
        conv1d_output = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        return self.layer_norm2(x + self.dropout(conv1d_output))

class MyModel(nn.Module):
    # 位置编码还得加！！！
    def __init__(self, 
                 dynamic_input_dim: int,    # 输入序列数据的维度
                 static_input_dim: int,     # 输入序列数据的维度
                 dynamic_embd_dim: int = 32, # 动态数据的embedding维度
                 static_embd_dim: int = 32,  # 静态数据的embedding维度
                 num_timestep: int = 8,     # 根据过去几个小时的数据预测
                 lead_time: int = 6,        # 预测未来几个消失的数据
                 num_head: int = 8,         # 多头注意力的头数
                 dropout=0.1) -> None:
        super().__init__()

        # 对数据进行embeding的网络
        self.dynamic_embd_net = nn.Sequential(
            nn.Linear(dynamic_input_dim, 80),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(80, self.dynamic_embd_dim)
        )

        # LSTM
        self.lstm = nn.LSTM(self.dynamic_embd_dim, self.lstm_hidden_dim, batch_first=True)

        # 生成预测数据的全连接网络
        self.pred_net = nn.Sequential(
            nn.Linear((self.lstm_hidden_dim + self.global_embd_dim) * self.num_timestep, 
                      self.lead_time),
            nn.ReLU(),
            nn.Linear(self.lead_time, self.lead_time),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Parameter:
         - x: (batch_size, seq_len, input_dim)
        Return: (batch_size, lead_time)
        """
        pass
    
    @staticmethod
    def NSE(y_hat, y):
        """计算一个 batch 的 NSE (Nash-Sutcliffe model efficiency coefficient)，
        自动忽略 y 全相同的样本，以避免数值不稳定。
        
        Parameters:
         - y_hat: (batch_size, seq_len)
         - y: (batch_size, seq_len)
        """
        mask = ~torch.isnan(y)
        y_hat = y_hat[mask]
        y = y[mask]
        
        eps = 1e-8  # 避免数值问题
        mean_y = torch.mean(y)
        denominator = ((y - mean_y) ** 2).sum()
        
        # 只计算有效 NSE
        if denominator < eps:
            return 0  # 返回 NaN 以指示无效值
        
        numerator = ((y_hat - y) ** 2).sum()
        value = 1 - (numerator / (denominator + eps))
        
        return float(value)

    
    @staticmethod
    def RMSE(y_hat, y):
        """计算一个batch的RMSE (Root Mean Square Error)
        Parameters:
         - y_hat: (batch_size, seq_len)
         - y: (batch_size, seq_len)
        """
        mask = ~torch.isnan(y)
        y = y[mask]             # (batch_size * seq_len)
        y_hat = y_hat[mask]

        value = torch.sqrt(((y - y_hat)**2).mean())
        return float(value)
    
    @staticmethod
    def MAE(y_hat, y):
        """计算一个batch的MAE (Mean Absolute Error)
        Parameters:
         - y_hat: (batch_size, seq_len)
         - y: (batch_size, seq_len)
        """
        mask = ~torch.isnan(y)
        y = y[mask]             # (batch_size * seq_len)
        y_hat = y_hat[mask]

        value = (y - y_hat).abs().mean()
        return float(value)
    
    @staticmethod
    def Bias(y_hat, y):
        """计算一个batch的Bias (Mean Bias Error)
        Parameters:
         - y_hat: (batch_size, seq_len)
         - y: (batch_size, seq_len)
        """
        mask = ~torch.isnan(y)
        y = y[mask]             # (batch_size * seq_len)
        y_hat = y_hat[mask]

        denominator = y.sum()
        numerator = (y_hat - y).sum()
        value = numerator / denominator
        return float(value)

if __name__ == '__main__':
    x = torch.randn(32, 8, 24)
    cond = torch.randn(32, 16)
    model = TemporalFiLM(24, 16, 8)
    print(model(x, cond).shape)