import torch

# 实际值 y 全为 5.0（方差为 0）
y = torch.tensor([5.0, 5.0, 5.0])  
# 预测值 y_hat 与 y 不同
y_hat = torch.tensor([6.0, 6.0, 6.0])  

# 计算 NSE
denominator = ((y - y.mean())**2).sum()  # 0.0
numerator = ((y_hat - y)**2).sum()       # 3.0
value = 1 - (numerator / denominator)    # 1 - (3.0 / 0.0) → 1 - inf → -inf
print(float(value))