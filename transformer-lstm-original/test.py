from model import HybirdModel
from hourly_data import DataInterface
from tqdm import tqdm
from loguru import logger
from utils import draw_with_metric

import numpy as np
import pandas as pd
import sys
import torch

if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}")

    model = HybirdModel(dynamic_input_dim=12, num_timestep=8, lead_time=6)
    model.load_state_dict(torch.load('./checkpoints/epoch5.pth', weights_only=True))
    
    datahub = DataInterface('Region_03_test.txt', '2005-10-01T00', '2006-09-30T00', default_num_workers=4)
    all_data_loader = datahub.get_data_loader('1990-10-01T00', '1991-09-30T00', batch_size=256, num_workers=2)

    device='cuda:0'
    model.eval()
    model.to(device)

    # # 在所有的流域上进行测试
    # with torch.no_grad():
    #     num_batch = 0.0
    #     avg_NSE = 0.0
    #     avg_RMSE = 0.0
    #     avg_MAE = 0.0
    #     avg_Bias = 0.0
    #     for batch in tqdm(all_data_loader, desc='Testing'):
    #         x = batch['x'].to(device)
    #         y = batch['y'].to(device)

    #         y_hat = model(x)
    #         avg_NSE += HybirdModel.NSE(y_hat, y)
    #         avg_RMSE += HybirdModel.RMSE(y_hat, y)
    #         avg_MAE += HybirdModel.MAE(y_hat, y)
    #         avg_Bias += HybirdModel.Bias(y_hat, y)
    #         num_batch += 1
    #     logger.info(f'Average NSE: {avg_NSE / num_batch: .4f} | Average RMSE: {avg_RMSE / num_batch: .4f} | Average MAE: {avg_MAE / num_batch: .4f} | Average Bias: {avg_Bias / num_batch: .4f}')

    # 在单个流域上进行测试
    NSEs = []
    RMSEs = []
    MAEs = []
    Biases = []
    for basin, loader in datahub:
        with torch.no_grad():
            num_batch = 0.0
            avg_NSE = 0.0
            avg_RMSE = 0.0
            avg_MAE = 0.0
            avg_Bias = 0.0
            for batch in tqdm(loader, desc=f'Testing {basin}'):
                x = batch['x'].to(device)
                y = batch['y'].to(device)

                y_hat = model(x)
                avg_NSE += HybirdModel.NSE(y_hat, y)
                avg_RMSE += HybirdModel.RMSE(y_hat, y)
                avg_MAE += HybirdModel.MAE(y_hat, y)
                avg_Bias += HybirdModel.Bias(y_hat, y)
                num_batch += 1
            
            logger.info(f'Average NSE: {avg_NSE / num_batch: .4f} | Average RMSE: {avg_RMSE / num_batch: .4f} | Average MAE: {avg_MAE / num_batch: .4f} | Average Bias: {avg_Bias / num_batch: .4f}')
            
            NSEs.append(avg_NSE / num_batch)
            RMSEs.append(avg_RMSE / num_batch)
            MAEs.append(avg_MAE / num_batch)
            Biases.append(avg_Bias / num_batch)
    draw_with_metric(datahub.basin_list, NSEs)