from model import HybirdModel
from hourly_data import DataInterface
from rich.progress import track

import torch

if __name__ == '__main__':
    model = HybirdModel(dynamic_input_dim=12, num_timestep=8, lead_time=6)

    model.load_state_dict(torch.load('./checkpoints/train_5y_10b/epoch15.pth', weights_only=True))
    datahub = DataInterface()
    loader = datahub.get_data_loader('1990-10-01T00', '1995-09-30T00', batch_size=256)

    device='cuda:0'
    model.eval()
    model.to(device)
    with torch.no_grad():
        num_batch = 0.0
        avg_NSE = 0.0
        for batch in track(loader, description='Testing'):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            y_hat = model(x)
            avg_NSE += HybirdModel.NSE(y_hat, y)
            num_batch += 1
        print(f'Average NSE: {avg_NSE / num_batch: .4f}')

# Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:04:21
# Average NSE:  0.8732