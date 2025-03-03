import torch

def NSE(y_hat, y, mean_y=None):
    """计算一个 batch 的 NSE (Nash-Sutcliffe model efficiency coefficient),
    自动忽略 y 全相同的样本，以避免数值不稳定。
    
    Parameters:
        - y_hat: (batch_size, seq_len)
        - y: (batch_size, seq_len)
        - mean_y: y 观测值的均值, 默认None表示逐Batch计算均值; 若提供, 则使用该值(适合整体NSE计算)
    """
    mask = ~torch.isnan(y)
    y_hat = y_hat[mask]
    y = y[mask]
    
    eps = 1e-8  # 避免数值问题
    if mean_y is None:
        mean_y = torch.mean(y)

    denominator = ((y - mean_y) ** 2).sum()
    if denominator < eps:   # 只计算有效 NSE
        return 0
    
    numerator = ((y_hat - y) ** 2).sum()
    value = 1 - (numerator / (denominator + eps))
    
    return float(value)

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