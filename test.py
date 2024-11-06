from model import MyModel
from data import DataInterface

import torch

model = MyModel(dynamic_input_dim=7,
                static_input_dim=27,
                hidden_dim=256)

model.load_state_dict(torch.load('./checkpoints/epoch35.pth', weights_only=True))
data = DataInterface()
loader = data.get_data_loader('1996-10-01', '1999-09-30', batch_size=1)

device='cpu'
model.eval()
with torch.no_grad():
    for batch in loader:
        x_h = batch['x_h'].to(device)
        x_f = batch['x_f'].to(device)
        x_s = batch['x_s'].to(device)
        y = batch['y'].to(device)

        x_h = x_h.transpose(0, 1)
        x_f = x_f.transpose(0, 1)

        y_hat = model(x_s, x_h, x_f)
        y_hat = MyModel.predict(y_hat)
        print(y)
        print(y_hat)
        input()
