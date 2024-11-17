import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

class HybirdModel(nn.Module):
    def __init__(self, dynamic_input_dim: int, num_timestep: int, lead_time: int, dropout=0.1) -> None:
        super().__init__()
        self.dynamic_embd_dim = 32          # 输入序列数据的维度
        self.num_timestep = num_timestep    # 根据过去几个小时的数据预测
        self.lead_time = lead_time          # 预测未来几个消失的数据
        self.num_head = 8                   # 多头自注意力的头数
        self.lstm_hidden_dim = 256          # LSTM的隐藏单元数目
        self.global_embd_dim = 128          # 最后与LSTM数据拼接的包含全局特征的向量的维度
        self.dropout = nn.Dropout(dropout)

        # 对数据进行embeding的网络
        self.dynamic_embd_net = nn.Sequential(
            nn.Linear(dynamic_input_dim, 80),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(80, self.dynamic_embd_dim)
        )

        # LSTM
        self.lstm = nn.LSTM(self.dynamic_embd_dim, self.lstm_hidden_dim, batch_first=True)

        # Multi-Head self Attention
        self.self_attention = nn.MultiheadAttention(self.lstm_hidden_dim, self.num_head, batch_first=True)

        # 用于多头自注意力后的残差连接的Layer Normalization
        self.layer_norm1 = nn.LayerNorm(self.lstm_hidden_dim)

        # 对应论文中的Weight Layer
        self.weight_layer_1 = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim),
            nn.ReLU()
        )

        # Weight的残差连接的Layer Normalization
        self.layer_norm2 = nn.LayerNorm(self.lstm_hidden_dim)

        # 残差连接后的Liner
        self.weight_layer_2 = nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim)

        # 一维卷积
        self.conv = nn.Conv1d(self.lstm_hidden_dim, self.lstm_hidden_dim, 3, padding='same')

        # 全局平均池化
        self.GAP = nn.AdaptiveAvgPool1d(output_size=1)

        # 卷积池化后的全连接层
        self.dense = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.global_embd_dim),
            nn.ReLU(),
            nn.Linear(self.global_embd_dim, self.global_embd_dim),
        )

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
        x = self.dynamic_embd_net(x)    # (batch_size, seq_len, embd_dim)
        lstm_output, _ = self.lstm(x)             # (batch_size, seq_len, lstm_embd_dim)
        x, _ = self.self_attention(lstm_output, lstm_output, lstm_output)    # (batch_size, seq_len, lstm_embd_dim)
        x = self.layer_norm1(x + self.dropout(x))
        x = self.weight_layer_1(x)
        x = self.weight_layer_2(x + self.dropout(x))
        # (batch_size, seq_len, lstm_embd_dim)

        x = self.conv(x.transpose(1, 2))    # (batch_size, dim, seq_len)
        x = self.GAP(x) # (batch_size, dim, 1)
        x = x.squeeze() # (batch_size, dim)

        x = self.dense(self.dropout(x))

        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        x = torch.concat((x.expand(-1, self.num_timestep,-1), lstm_output), dim=2)

        return self.pred_net(x.flatten(1))
    
    @staticmethod
    def NSE(y_hat, y):
        """计算一个batch的七天的NSE (Nash-Sutcliffe model efficiency coefficient)
        Parameters:
         - y_hat: (batch_size, seq_len)
         - y: (batch_size, seq_len)
        """
        mask = ~torch.isnan(y)
        y = y[mask]             # (batch_size * seq_len)
        y_hat = y_hat[mask]     # (batch_size * seq_len)

        denominator = ((y - y.mean())**2).sum()
        numerator = ((y_hat - y)**2).sum()
        value = 1 - (numerator / denominator)
        return float(value)

if __name__ == '__main__':
    model = HybirdModel(dynamic_input_dim=11, num_timestep=8, lead_time=6)
    x = torch.randn(32, 8, 11)
    print(model(x).shape)