from model import MyModel
from hourly_data import DataInterface
from tqdm import tqdm
from loguru import logger
from utils import draw_with_metric
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sys
import torch

def test_by_basin(model, datahub, basin, device='cuda:0'):
    """
    可视化指定流域的真实值 (y) 和预测值 (y_hat)
    
    :param model: 训练好的模型
    :param datahub: 数据接口对象
    :param basin: 指定的流域 ID
    :param device: 计算设备，默认 'cuda:0'
    """
    model.eval()
    model.to(device)
    
    loader = datahub.get_loader_by_basin(basin)
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Testing {basin}'):
            x_d = batch['x_d'].to(device)
            x_s = batch['x_s'].to(device)
            y = batch['y'].to(device)
            
            y_hat = model(x_d, x_s)
            
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())
    
    y_true = np.concatenate(y_true_list, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted', color='red', linestyle='dashed', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Streamflow')
    plt.title(f'Basin {basin} - True vs Predicted Streamflow')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}")

    model = MyModel(12, 27, num_timestep=8, lead_time=6)
    model.load_state_dict(torch.load('./checkpoints/epoch20.pth', weights_only=True))
    
    datahub = DataInterface('Region_03_test.txt', '2005-10-01T00', '2006-09-30T00', default_num_workers=4)
    all_data_loader = datahub.get_data_loader('1990-10-01T00', '1991-09-30T00', batch_size=256, num_workers=2)

    test_by_basin(model, datahub, '02427250')
    test_by_basin(model, datahub, '02198100')
    test_by_basin(model, datahub, '02469800')

    # device='cuda:0'
    # model.eval()
    # model.to(device)

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
    # NSEs = []
    # RMSEs = []
    # MAEs = []
    # Biases = []
    # y_hat_list = []
    # y_list = []
    # for basin, loader in datahub:
    #     with torch.no_grad():
    #         num_batch = 0.0
    #         avg_NSE = 0.0
    #         avg_RMSE = 0.0
    #         avg_MAE = 0.0
    #         avg_Bias = 0.0
    #         for batch in tqdm(loader, desc=f'Testing {basin}'):
    #             x_d = batch['x_d'].to(device)
    #             x_s = batch['x_s'].to(device)
    #             y = batch['y'].to(device)

    #             y_hat = model(x_d, x_s)
    #             avg_NSE += MyModel.NSE(y_hat, y, datahub.basins_mean[basin])
    #             avg_RMSE += MyModel.RMSE(y_hat, y)
    #             avg_MAE += MyModel.MAE(y_hat, y)
    #             avg_Bias += MyModel.Bias(y_hat, y)
    #             num_batch += 1

    #             y_hat_list.append(y_hat.cpu().numpy())
    #             y_list.append(y.cpu().numpy())
            
    #         logger.info(f'Average NSE: {avg_NSE / num_batch: .4f} | Average RMSE: {avg_RMSE / num_batch: .4f} | Average MAE: {avg_MAE / num_batch: .4f} | Average Bias: {avg_Bias / num_batch: .4f}')
            
    #         NSEs.append(avg_NSE / num_batch)
    #         RMSEs.append(avg_RMSE / num_batch)
    #         MAEs.append(avg_MAE / num_batch)
    #         Biases.append(avg_Bias / num_batch)
    # draw_with_metric(datahub.basin_list, NSEs)