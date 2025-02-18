# This file includes code from the NeuralHydrology project, licensed under the BSD 3-Clause License.
# Copyright (c) 2021, NeuralHydrology. See the LICENSE file for details.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from torchinfo import summary

class CMAL(nn.Module):
    """Countable Mixture of Asymmetric Laplacians(CMAL)
    修改自 neuralhydrology.modelzoo.head.CAML 
    详见: https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.modelzoo.head.html#neuralhydrology.modelzoo.head.CMAL

    Parameters:
     - n_in: 输入特征的数目, 之前模块的输出维度
     - num_target: 目标特征的数目, 例如如果做流量预测, 那么只有一个目标变量, 即QObs(mm/d)
     - num_distribution: 由几个概率分布混合而成, 这里默认采用三个
     - n_hidden: 隐藏层神经元数目

    ``n_out = num_target * num_distribution * 4``
    由于Asymmetric Laplacians由三个参数决定, 而CMAL由多个概率分布决定, 因此还需为每个分布
    设置一个权重参数, 故每个概率分布实际需要四个参数, 分别是:
     - mu: ALD的均值
     - b: ALD的尺度参数
     - tau: ALD的偏度参数
     - p: 该分布的权重

    """
    def __init__(self, n_in: int, num_target: int = 1, num_distribution: int = 3, n_hidden: int = 100):
        super(CMAL, self).__init__()
        n_out = num_target * num_distribution * 4
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self._softplus = torch.nn.Softplus(2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = self._softplus(b_latent) + self._eps  # scale > 0 (softplus was working good in tests)
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps  # 0 > tau > 1
        p = (1 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps  # sum(pi) = 1 & pi > 0

        return {'mu': m, 'b': b, 'tau': t, 'pi': p}

class MyModel(nn.Module):
    def __init__(self, dynamic_input_dim, static_input_dim, hidden_dim, hindcast_length = 336, forecast_horizon = 6):
        super(MyModel, self).__init__()
        self.dynamic_input_dim = dynamic_input_dim      # 动态输入维度
        self.dynamic_embd_dim = 32                      # 动态嵌入维度
        self.static_input_dim = static_input_dim        # 静态输入维度
        self.static_embd_dim = 32                       # 静态嵌入维度
        self.hidden_dim = hidden_dim                    # Encoder & Decoder隐藏维度
        self.hindcast_length = hindcast_length          # 后顾时间长度
        self.forecast_horizon = forecast_horizon        # 预测时间长度
        # 状态转移网络分别是线性层和带激活函数的线性层,没有隐藏层,因此不设置状态转移网络的隐藏层维度.

        # 对静态数据进行嵌入, 静态数据维度较多, 因此通过嵌入降维
        self.static_embd_net = nn.Sequential(
            nn.Linear(self.static_input_dim, 80),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(80, self.static_embd_dim)
        )

        # 对动态数据进行嵌入, 动态数据维度较少, 因此通过嵌入增维
        self.dynamic_embd_net = nn.Sequential(
            nn.Linear(self.dynamic_input_dim, 64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(64, self.dynamic_embd_dim)
        )

        # 后顾(Hindcast) LSTM
        self.hindcast_lstm = nn.LSTM(
            input_size=self.dynamic_embd_dim + self.static_embd_dim, 
            hidden_size=self.hidden_dim
        )

        # 预测(Forecast) LSTM
        self.forecast_lstm = nn.LSTM(
            input_size=self.dynamic_embd_dim + self.static_embd_dim, 
            hidden_size=self.hidden_dim
        )

        # Cell State Transfer Network
        self.cell_transfer_net = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Hidden State Transfer Network
        self.hidden_transfer_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )

        # CMAL Head
        # Countable Mixture of Asymmetric Laplacians (CAML)
        # 作用是将预测输出映射到CMAL参数
        self.cmal_head = CMAL(
            n_in=self.hidden_dim,
            num_target=1,
            num_distribution=3 
            )
    
    def forward(self, x_s: torch.Tensor, x_h: torch.Tensor, x_f: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters:
         - x_s: 静态特征(static), (batch_size, dim)
         - x_h: 后顾动态特征(hindcast), (seq_len, batch_size, dim)
         - x_f: 预测动态特征(forecast), (seq_len, batch_size, dim)
        """
        batch_size = x_h.shape[1]

        # 对输入数据进行embedding
        x_s = self.static_embd_net(x_s)     # (batch_size, 32)
        x_h = self.dynamic_embd_net(x_h)    # (seq_len, batch_size, 32)
        x_f = self.dynamic_embd_net(x_f)    # (seq_len, batch_size, 32)

        # 将静态数据拼接到动态数据中
        x_h = torch.cat((x_s.unsqueeze(0).expand(self.hindcast_length, -1, -1), x_h), dim=-1)   # (seq_len, batch_size, 64)

        x_f = torch.cat((x_s.unsqueeze(0).expand(self.forecast_horizon, -1, -1), x_f), dim=-1)   # (seq_len, batch_size, 64)

        # 后顾网络对历史数据进行处理
        hindcast_output, (h_hindcast_op, c_hindcast_op) = self.hindcast_lstm(
            x_h,
            (torch.zeros(1, batch_size, self.hidden_dim, device=x_h.device), torch.zeros(1, batch_size, self.hidden_dim, device=x_h.device))
        )   # h & c : (1, batch_size, hidden_dim)

        # 状态转移网络处理
        h_forecast_in = self.hidden_transfer_net(h_hindcast_op)
        c_forecast_in = self.cell_transfer_net(c_hindcast_op)

        # 预测网络进行预测
        forecast_output, (h_forecast_op, c_forecast_op) = self.forecast_lstm(
            x_f,
            (h_forecast_in, c_forecast_in)
        )
        # forecast_output: (seq_len, batch_size, hidden_dim)
        # h_forecast_op: (1, batch_size, hidden_dim)
        # c_forecast_op: (1, batch_size, hidden_dim)

        forecast_output = forecast_output.transpose(0, 1)   # (batch_size, seq_len, hidden_dim)
        return self.cmal_head(forecast_output)  # (batch_size, seq_len, num_distribution)
    
    @staticmethod
    def calculate_loss(y_hat: Dict[str, torch.Tensor], y: torch.Tensor):
        """计算损失
        Parameters:
         - y_hat: (batch_size, seq_len, num_distribution)
         - y: (batch_size, seq_len)
        """
        eps = 1e-5
        mask = ~torch.isnan(y)
        y = y[mask].unsqueeze(1)    # (batch_size * seq_len, 1)
        m = y_hat['mu'][mask]       # (batch_size * seq_len, num_distribution) (no limit)
        b = y_hat['b'][mask]        # (batch_size * seq_len, num_distribution) (>0)
        t = y_hat['tau'][mask]      # (batch_size * seq_len, num_distribution) (0-1)
        p = y_hat['pi'][mask]       # (batch_size * seq_len, num_distribution) (0-1)

        error = y - m               # (batch_size * seq_len, num_distribution)
        log_like = torch.log(t) + \
                   torch.log(1.0 - t) - \
                   torch.log(b) - \
                   torch.max(t * error, (t - 1.0) * error) / b
        log_weights = torch.log(p + eps)    # (batch_size * seq_len, num_distribution)

        result = torch.logsumexp(log_weights + log_like, dim=1) # batch_size * seq_len
        result = -torch.mean(result)
        return result
    
    @staticmethod
    def predict(y_hat: Dict[str, torch.Tensor]):
        """使用 权重*均值 计算一个batch的所有forecast horizon的预测值 
        Parameter:
         - y_hat: 有'mu', 'b', 'tau', 'pi'四个键
        """
        m = y_hat['mu'].detach()
        p = y_hat['pi'].detach()
        pred = (m * p).sum(dim=-1)
        return pred     # batch_size seq_len
    
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
    model = MyModel(dynamic_input_dim=12,
                static_input_dim=27,
                hidden_dim=256)
    summary(model, input_size=[(64, 27), (336, 64, 12), (6, 64, 12)])