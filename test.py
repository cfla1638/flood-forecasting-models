import torch

a = torch.randn(2, 6)
a[0, 0] = float('nan')
mask = ~torch.isnan(a)
print(a)
print(a[mask])  # tensor([ 0.0000, -0.0732,  0.0000, -0.0732,  0.0000, -0.0732,  0.0000, -0.0732])