from model import MyModel
from hourly_data import DataInterface
from tqdm import tqdm
from loguru import logger
from utils import draw_with_metric, plot_predictions
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
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    logger.add("./log/test_log{time}.log", level="INFO", rotation="20 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

class TestInterface(object):
    def __init__(self, opts) -> None:
        self.opts = opts

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
        
        NSEs = np.array(NSEs)
        NSEs[NSEs < -1] = -1
        logger.info(f'Average NSE: {np.mean(NSEs): .4f} | Average RMSE: {np.mean(RMSEs): .4f} | Average MAE: {np.mean(MAEs): .4f} | Average Bias: {np.mean(Biases): .4f}')

        draw_with_metric(datahub.basin_list, NSEs)
        plot_predictions(np.concatenate(y_list, axis=0).reshape(-1), np.concatenate(y_hat_list, axis=0).reshape(-1))

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
        y_true = np.concatenate(y_true_list, axis=0)    # (seq_len, 6) 需要展示 [:,0]
        y_pred = np.concatenate(y_pred_list, axis=0)    # (seq_len, 6) 需要展示 [:,0], [1:,1], [2:,2], [3:,3] ...
        
        # 绘图
        seq_len, lead_time = y_true.shape
        lead_time = 6
        colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:, 0], label='观测值', color='blue')
        for i in range(lead_time):
            plt.plot(range(i, seq_len), y_pred[i:, i], label=f'提前{i+1}步预测', color=colors[i], linestyle='dashed', alpha=(lead_time-i)/lead_time)
        plt.xlabel('时间步')
        plt.ylabel('流量值')
        plt.title(f'流域 {basin} - 观测值与预测值对比')
        plt.legend()
        plt.show()

        # 绘制预测值与实际值的散点图
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        plot_predictions(y_true, y_pred)

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

        datahub = DataInterface(self.opts.basin_list, self.opts.start_time, self.opts.end_time, default_batch_size=opts.batch_size, default_num_workers=opts.num_workers, dynamic_meanstd=opts.dynamic_meanstd, static_meanstd=opts.static_meanstd)

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

# python -u test.py --use_GPU --GPU_id 0 --num_workers=4 --start_time=2009-10-01T00 --end_time=2011-09-30T00 --model_path=./checkpoints/epoch3.pth --basin_list=30_basin_list_evenly.txt --test_for_single_basin --gauge_id=12488500

# python -u test.py --use_GPU --GPU_id 0 --num_workers=4 --start_time=2009-10-01T00 --end_time=2011-09-30T00 --dynamic_meanstd=dynamic_30_basin_list_evenly.csv --model_path=./checkpoints/epoch3.pth --basin_list=30_basin_list_evenly_test.txt --test_basin_by_basin

