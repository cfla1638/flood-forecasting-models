# GRU模型
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from torchinfo import summary
from loguru import logger

class MyModel(nn.Module):
    def __init__(self, 
                 dynamic_input_dim: int,    # 输入序列数据的维度
                 static_input_dim: int,     # 输入序列数据的维度
                 dynamic_embd_dim: int = 256, # 动态数据的embedding维度
                 static_embd_dim: int = 32,  # 静态数据的embedding维度
                 num_timestep: int = 8,     # 根据过去几个小时的数据预测
                 lead_time: int = 6,        # 预测未来几个消失的数据
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

        hidden_dim = dynamic_embd_dim + static_embd_dim
        # GRU
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_timestep, 64),
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

        x_s = x_s.unsqueeze(1).expand(-1, x_d.shape[1], -1)  # (batch_size, seq_len, static_embd_dim)
        x_d = torch.concat([x_d, x_s], dim=-1)  # (batch_size, seq_len, dynamic_embd_dim + static_embd_dim)

        # LSTM
        x_d, _ = self.gru(x_d)              # (batch_size, seq_len, dynamic_embd_dim)

        return self.output_layer(x_d.flatten(1))    # (batch_size, lead_time)
    
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
    # x_d = torch.randn(32, 8, 12)
    # x_s = torch.randn(32, 27)
    model = MyModel(12, 27)
    model.apply(init_weights)
    # y = model(x_d, x_s)
    # print(y.shape)
    summary(model, input_size=[(32, 8, 12), (32, 27)], device='cpu')
