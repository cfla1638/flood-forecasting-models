import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from torchinfo import summary
from loguru import logger

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
        conv1d_output = F.leaky_relu(conv1d_output, negative_slope=0.1)
        return self.layer_norm2(x + self.dropout(conv1d_output))

class MyModel(nn.Module):
    def __init__(self, 
                 dynamic_input_dim: int,    # 输入序列数据的维度
                 static_input_dim: int,     # 输入序列数据的维度
                 dynamic_embd_dim: int = 128, # 动态数据的embedding维度
                 static_embd_dim: int = 128,  # 静态数据的embedding维度
                 num_timestep: int = 8,     # 根据过去几个小时的数据预测
                 lead_time: int = 6,        # 预测未来几个消失的数据
                 num_head: int = 8,         # 多头注意力的头数
                 encoder_layers: int = 3,   # 编码器的层数
                 dropout=0.1) -> None:
        super().__init__()

        # 对动态数据进行embeding的网络
        self.dynamic_embd_net = nn.Sequential(
            nn.Linear(dynamic_input_dim, dynamic_embd_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(dynamic_embd_dim, dynamic_embd_dim)
        )

        # 对静态数据进行embeding的网络
        self.static_embd_net = nn.Sequential(
            nn.Linear(static_input_dim, static_embd_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(static_embd_dim, static_embd_dim)
        )

        # GRU
        self.gru = nn.GRU(input_size=dynamic_embd_dim, hidden_size=dynamic_embd_dim, num_layers=2, batch_first=True, dropout=0.2)

        # Encoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(dim=dynamic_embd_dim, num_head=num_head, dropout=dropout) for _ in range(encoder_layers)
        ])

        # TemporalFiLM Layers
        self.temporal_film_layers = nn.ModuleList([
            TemporalFiLM(dynamic_embd_dim, static_embd_dim, num_timestep) for _ in range(encoder_layers)
        ])

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(dynamic_embd_dim * num_timestep, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, lead_time)
        )


    def forward(self, x_d : torch.Tensor, x_s : torch.Tensor)-> torch.Tensor:
        """
        Parameter:
         - x_d: (batch_size, seq_len, dynamic_input_dim) 动态数据
         - x_s: (batch_size, static_input_din) 静态数据
        Return: (batch_size, lead_time)
        """
        x_d = self.dynamic_embd_net(x_d)    # (batch_size, seq_len, dynamic_embd_dim)
        x_s = self.static_embd_net(x_s)     # (batch_size, static_embd_dim)

        # GRU
        x_d, _ = self.gru(x_d)              # (batch_size, seq_len, dynamic_embd_dim)

        # Encoder & TemporalFiLM Layers
        for encoder, film in zip(self.encoder_layers, self.temporal_film_layers):
            x_d = encoder(x_d)              # (batch_size, seq_len, dynamic_embd_dim)
            x_d = film(x_d, x_s)            # (batch_size, seq_len, dynamic_embd_dim)

        # Output Layer
        return self.output_layer(x_d.flatten(1))       # (batch_size, lead_time)
    
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        elif isinstance(m, nn.MultiheadAttention):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

if __name__ == '__main__':
    x_d = torch.randn(32, 8, 12)
    x_s = torch.randn(32, 27)
    model = MyModel(12, 27)
    model.apply(init_weights)
    y = model(x_d, x_s)
    print(y.shape)
