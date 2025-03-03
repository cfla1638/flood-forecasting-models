from model import MyModel
from hourly_data import DataInterface
from tqdm import tqdm
from loguru import logger
from utils import draw_with_metric
from args import Args
from metrics import NSE, RMSE, MAE, Bias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch

def setup_logger():
    # 设置logger
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}")
    # logger.add("./log/log{time}.log", level="INFO", rotation="20 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

class TestInterface(object):
    def __init__(self, opts) -> None:
        self.opts = opts

    @staticmethod
    def _test_for_all_basins(model, datahub, start_time, end_time, batch_size, num_workers, device='cuda:0'):
        logger.info('Testing for all basins')
        
        loader = datahub.get_data_loader(start_time, end_time, batch_size=batch_size, num_workers=num_workers)
        # 在所有的流域上进行测试
        with torch.no_grad():
            num_batch = 0.0
            avg_NSE = 0.0
            avg_RMSE = 0.0
            avg_MAE = 0.0
            avg_Bias = 0.0
            for batch in tqdm(loader, desc='Testing'):
                x_d = batch['x_d'].to(device)
                x_s = batch['x_s'].to(device)
                y = batch['y'].to(device)

                y_hat = model(x_d, x_s)
                avg_NSE += NSE(y_hat, y)
                avg_RMSE += RMSE(y_hat, y)
                avg_MAE += MAE(y_hat, y)
                avg_Bias += Bias(y_hat, y)
                num_batch += 1
            logger.info(f'Average NSE: {avg_NSE / num_batch: .4f} | Average RMSE: {avg_RMSE / num_batch: .4f} | Average MAE: {avg_MAE / num_batch: .4f} | Average Bias: {avg_Bias / num_batch: .4f}')

    @staticmethod
    def _test_basin_by_basin(model, datahub, device='cuda:0'):
        logger.info('Testing basin by basin')

        NSEs = []
        RMSEs = []
        MAEs = []
        Biases = []
        y_hat_list = []
        y_list = []

        total_iter_cnt = len(datahub.basin_list)
        for cur, (basin, loader) in enumerate(datahub):
            with torch.no_grad():
                num_batch = 0.0
                avg_NSE = 0.0
                avg_RMSE = 0.0
                avg_MAE = 0.0
                avg_Bias = 0.0
                for batch in tqdm(loader, desc=f'Testing {basin} [{cur + 1}/{total_iter_cnt}]'):
                    x_d = batch['x_d'].to(device)
                    x_s = batch['x_s'].to(device)
                    y = batch['y'].to(device)

                    y_hat = model(x_d, x_s)
                    avg_NSE += NSE(y_hat, y, datahub.basins_mean[basin])
                    avg_RMSE += RMSE(y_hat, y)
                    avg_MAE += MAE(y_hat, y)
                    avg_Bias += Bias(y_hat, y)
                    num_batch += 1

                    y_hat_list.append(y_hat.cpu().numpy())
                    y_list.append(y.cpu().numpy())
                
                logger.info(f'Average NSE: {avg_NSE / num_batch: .4f} | Average RMSE: {avg_RMSE / num_batch: .4f} | Average MAE: {avg_MAE / num_batch: .4f} | Average Bias: {avg_Bias / num_batch: .4f}')
                
                NSEs.append(avg_NSE / num_batch)
                RMSEs.append(avg_RMSE / num_batch)
                MAEs.append(avg_MAE / num_batch)
                Biases.append(avg_Bias / num_batch)
        draw_with_metric(datahub.basin_list, NSEs)

    @staticmethod
    def _test_for_single_basin(model, datahub, basin, device='cuda:0'):
        """
        可视化指定流域的真实值 (y) 和预测值 (y_hat)
        
        :param model: 训练好的模型
        :param datahub: 数据接口对象
        :param basin: 指定的流域 ID
        :param device: 计算设备，默认 'cuda:0'
        """
        logger.info(f'Testing for single basin {basin}')

        loader = datahub.get_loader_by_basin(basin)
        y_true_list = []
        y_pred_list = []
        num_batch = 0.0
        avg_NSE = 0.0
        avg_RMSE = 0.0
        avg_MAE = 0.0
        avg_Bias = 0.0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Testing {basin}'):
                x_d = batch['x_d'].to(device)
                x_s = batch['x_s'].to(device)
                y = batch['y'].to(device)
                
                y_hat = model(x_d, x_s)

                avg_NSE += NSE(y_hat, y, datahub.basins_mean[basin])
                avg_RMSE += RMSE(y_hat, y)
                avg_MAE += MAE(y_hat, y)
                avg_Bias += Bias(y_hat, y)
                num_batch += 1
                
                y_true_list.append(y.cpu().numpy())
                y_pred_list.append(y_hat.cpu().numpy())
        logger.info(f'Average NSE: {avg_NSE / num_batch: .4f} | Average RMSE: {avg_RMSE / num_batch: .4f} | Average MAE: {avg_MAE / num_batch: .4f} | Average Bias: {avg_Bias / num_batch: .4f}')
        y_true = np.concatenate(y_true_list, axis=0).reshape(-1)
        y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1)
        
        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='True', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted', color='red', linestyle='dashed', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Streamflow')
        plt.title(f'Basin {basin} - Observerd vs Predicted Streamflow')
        plt.legend()
        plt.show()

    def main(self):
        opts = self.opts

        model = MyModel(12, 27, num_timestep=8, lead_time=6)
        model.load_state_dict(torch.load(self.opts.model_path, weights_only=True))
        logger.info(f'Model loaded from {self.opts.model_path}')

        if opts.use_GPU:
            device = f'cuda:{opts.GPU_id}'
        else:
            device = 'cpu'
        logger.info(f'Testing on {device}')

        model.eval()
        model.to(device)

        datahub = DataInterface(self.opts.basins_list, self.opts.start_time, self.opts.end_time, default_batch_size=opts.batch_size, default_num_workers=opts.num_workers)

        if opts.test_for_all_basins:
            self._test_for_all_basins(model, datahub, self.opts.start_time, self.opts.end_time, opts.batch_size, opts.num_workers, device=device)

        if opts.test_basin_by_basin:
            self._test_basin_by_basin(model, datahub, device=device)

        if opts.test_for_single_basin:
            self._test_for_single_basin(model, datahub, opts.gauge_id, device=device)

if __name__ == '__main__':
    args = Args()
    args.set_test_args()
    setup_logger()
    test_interface = TestInterface(args.get_opts())
    test_interface.main()

# python -u test.py --use_GPU --GPU_id 0 --num_workers=4 --start_time=2005-10-01T00 --end_time=2007-09-30T00 --model_path=./checkpoints/epoch9.pth --basins_list=32_basin_list.txt --test_basin_by_basin