from model import MyModel
from hourly_data import DataInterface
from rich.progress import track

import torch

if __name__ == '__main__':
    model = MyModel(dynamic_input_dim=12,
                static_input_dim=27,
                hidden_dim=256)

    model.load_state_dict(torch.load('./checkpoints/epoch40.pth', weights_only=True))
    datahub = DataInterface()
    loader = datahub.get_data_loader('1990-10-01T00', '1995-09-30T00', batch_size=256)

    device='cuda:0'
    model.eval()
    model.to(device)
    with torch.no_grad():
        num_batch = 0.0
        avg_NSE = 0.0
        for batch in track(loader, description='Testing'):
            x_h = batch['x_h'].to(device)
            x_f = batch['x_f'].to(device)
            x_s = batch['x_s'].to(device)
            y = batch['y'].to(device)

            x_h = x_h.transpose(0, 1)
            x_f = x_f.transpose(0, 1)

            y_hat = model(x_s, x_h, x_f)
            y_hat = MyModel.predict(y_hat)
            avg_NSE += MyModel.NSE(y_hat, y)
            num_batch += 1
        print(f'Average NSE: {avg_NSE / num_batch: .4f}')

# Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:40
# Average NSE:  0.9097