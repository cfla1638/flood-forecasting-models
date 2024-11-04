# from pathlib import Path
# from neuralhydrology.utils.-config import Config
# from neuralhydrology.datasetzoo.camelsus import CamelsUS
# from torch.utils.data import DataLoader
# from neuralhydrology.nh_run import start_run
# from datetime import datetime, date
import numpy as np

# config_file = Path('./data/config/1_basin.yml')
# config = Config(config_file)

# dataset = CamelsUS(config, is_train=True, period='train')

# # loader = DataLoader(dataset, 256, True, collate_fn=dataset.collate_fn)
# # for i in loader:
# #     print(i['x_d'].shape)
# #     input()

# # print(len(dataset))
# # print(dataset[0]['x_d'].shape)
# # print(dataset[0]['y'].shape)
# for i in range(len(dataset)):
#     print(dataset[i].keys())
#     input()
# # print(dataset[1234]['x_d'])
a = np.datetime64('2024-09-20')
b = np.datetime64('2024-09-25')
print((b-a).astype(np.int32))
