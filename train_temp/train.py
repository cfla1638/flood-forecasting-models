from model import MyModel, init_weights
from hourly_data import DataInterface
from tqdm import tqdm
from pathlib import Path
from args import Args
from loguru import logger
from metrics import NSE, RMSE, MAE, Bias

import os
import sys
import torch
import settings

def setup_logger():
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    logger.add("./log/train_log{time}.log", level="INFO", rotation="20 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

class TrainInterface(object):
    def __init__(self, opts) -> None:
        self.opts = opts

    @staticmethod
    def _train_epoch(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion, epoch: int, device: str):
        """进行一个epoch的训练, 打印本轮训练的平均损失
        Parameters:
         - model: 要训练的模型
         - train_loader: 加载训练数据, Dataloader的子类
         - optimizer: 优化器
         - criterion: 损失函数
         - epoch: 当前在进行的epoch的编号
         - device: 训练设备, 例如'cpu', 'cuda:0'
        """
        model.train()
        model.to(device)
        optimizer.zero_grad()

        num_batch = 0.0
        losssum = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            # 在加载数据时将数据转移到Device上
            x_d = batch['x_d'].to(device)
            x_s = batch['x_s'].to(device)
            y = batch['y'].to(device)

            y_hat = model(x_d, x_s)
            loss = criterion(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()

            losssum += loss.detach().cpu().numpy().item()
            num_batch += 1
        logger.info(f'Epoch {epoch} | average loss = {losssum / num_batch : .4f}')

    
    @staticmethod
    def _validate(model: torch.nn.Module, val_loader:torch.utils.data.DataLoader, device: str):
        """训练的过程中进行验证
        Parameters:
         - model: 要验证的模型
         - val_loader: 加载训练数据, Dataloader的子类
         - device: 训练设备, 例如'cpu', 'cuda:0'
        """
        logger.info('Start validating.')
        model.eval()

        num_batch = 0.0
        avg_NSE = 0.0
        avg_RMSE = 0.0
        avg_MAE = 0.0
        avg_Bias = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'validating'):
                # 在加载数据时将数据转移到Device上
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
    def _save_model(model: torch.nn.Module, epoch, save_dir: Path):
        """保存模型, 只保存参数
        Parameters:
         - model: 要保存模型
         - epoch: 当前的epoch编号
         - save_dir: 保存到哪个文件夹
        """
        save_dir = Path(save_dir)
        model_name = f'epoch{epoch}.pth'
        save_path = save_dir / model_name
        torch.save(model.state_dict(), save_path)
        logger.info(f'Model {model_name} saved.')

    def main(self):
        """训练的流程
        """
        opts = self.opts
        if not os.path.exists(opts.checkpoints_dir):
            os.mkdir(opts.checkpoints_dir)
            logger.info(f'Create checkpoints dir {opts.checkpoints_dir}')
        
        train_data_interface = DataInterface(opts.train_basin_list, dataset_path=opts.train_dataset_path)
        val_data_interface = DataInterface(opts.val_basin_list, dynamic_meanstd=opts.dynamic_meanstd, static_meanstd=opts.static_meanstd, dataset_path=opts.val_dataset_path)

        if not opts.validate:
            train_loader = train_data_interface.get_data_loader(opts.train_start_time, opts.train_end_time, opts.batch_size, num_workers=opts.num_workers)
        val_loader = val_data_interface.get_data_loader(opts.val_start_time, opts.val_end_time, opts.batch_size, num_workers=opts.num_workers)

        model = MyModel(12, 27, num_timestep=8, lead_time=6)
        model.apply(init_weights)
        logger.info('Model initialized.')

        # 检查是否要加载预训练的模型
        if opts.pretrain is not None:
            model.load_state_dict(torch.load(opts.pretrain, weights_only=True))
            logger.info(f'Load pretrain model from {opts.pretrain}')

        if opts.use_GPU:
            device = f'cuda:{opts.GPU_id}'
        else:
            device = 'cpu'

        if opts.validate:
            model.to(device)
            self._validate(model, val_loader, device)
            return
        
        optimizer = torch.optim.Adam(model.parameters())
        Loss = torch.nn.MSELoss()
        logger.info('Train on ' + device)
        logger.info('Start training.')
        for epoch in range(opts.start_epoch, opts.epoch + 1):
            self._train_epoch(model, train_loader, optimizer, Loss, epoch, device)
            # 每save_freq保存一次模型, 训练的最后一个epoch也需要保存模型
            if epoch % opts.save_freq == 0 or epoch == opts.epoch:
                self._save_model(model, epoch, opts.checkpoints_dir)

            # 检查是否需要验证
            if opts.val_freq is not None and epoch % opts.val_freq == 0:
                self._validate(model, val_loader, device)

if __name__ == '__main__':
    args = Args()
    args.set_train_args()
    setup_logger()
    train_interface = TrainInterface(args.get_opts())
    train_interface.main()

# python train.py --batch_size=256 --train_start_time=2002-10-01T00 --train_end_time=2003-09-30T00 --epoch=50 --save_freq=1 --use_GPU --GPU_id=0 --val_freq=1 --val_start_time=2007-10-01T00 --val_end_time=2009-09-30T00 --num_workers=8 --train_basin_list=150_basin_list_evenly.txt --val_basin_list=30_basin_list_evenly_test.txt --train_dataset_path=../data/CAMELS_US/hourly/150_basin_list_evenly_1y.nc --val_dataset_path=../data/CAMELS_US/hourly/30_basin_list_evenly_test_val_2y.nc --dynamic_meanstd=dynamic_150_basin_list_evenly.csv --static_meanstd=static_150_basin_list_evenly.csv

# python train.py --batch_size=256 --train_start_time=2002-10-01T00 --train_end_time=2007-09-30T00 --epoch=50 --save_freq=1 --use_GPU --GPU_id=0 --val_freq=1 --val_start_time=2007-10-01T00 --val_end_time=2009-09-30T00 --num_workers=8 --train_basin_list=30_basin_list_evenly.txt --val_basin_list=30_basin_list_evenly.txt --train_dataset_path=../data/CAMELS_US/hourly/30_basin_list_evenly_train_5y.nc --val_dataset_path=../data/CAMELS_US/hourly/30_basin_list_evenly_val_2y.nc --dynamic_meanstd=dynmaic_30_basin_list_evenly.csv --static_meanstd=static_30_basin_list_evenly.csv
