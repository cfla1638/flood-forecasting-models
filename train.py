from model import MyModel
from data import DataInterface
from tqdm import tqdm

import torch
import settings

model1 = MyModel(dynamic_input_dim=7,
                static_input_dim=27,
                hidden_dim=256)

data_interface = DataInterface()
train_loader = data_interface.get_data_loader('1990-01-01', '1999-01-01')
test_loader = data_interface.get_data_loader('1985-01-01', '1989-01-01', batch_size=256)

# model.to(settings.device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 20
# # 训练循环
# for epoch in range(num_epochs):
#     cnt = 0.0
#     avgloss = 0.0
#     for batch in tqdm(train_loader, desc=f'epoch {epoch}'):
#         x_h = batch['x_h'].to(settings.device)
#         x_f = batch['x_f'].to(settings.device)
#         x_s = batch['x_s'].to(settings.device)
#         y = batch['y'].to(settings.device)

#         x_h = x_h.transpose(0, 1)
#         x_f = x_f.transpose(0, 1)

#         # 前向传播
#         y_hat = model(x_s, x_h, x_f)
#         loss = model.calculate_loss(y_hat, y)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         optimizer.zero_grad()
#         avgloss += loss.detach().cpu().numpy().item()
#         cnt += 1
#     print(avgloss / cnt)

# torch.save(model, './data/model/eopch20.model')

model = torch.load('./data/model/eopch10.model')
model.eval()

with torch.no_grad():
    for batch in test_loader:
        x_h = batch['x_h'].to(settings.device)
        x_f = batch['x_f'].to(settings.device)
        x_s = batch['x_s'].to(settings.device)
        y = batch['y'].to(settings.device)

        x_h = x_h.transpose(0, 1)
        x_f = x_f.transpose(0, 1)

        y_hat = model(x_s, x_h, x_f)
        y_hat = model.predict(y_hat)
        print(model1.NSE(y_hat, y))
        input()