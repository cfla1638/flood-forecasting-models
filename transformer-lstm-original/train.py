from model import HybirdModel
from hourly_data import DataInterface
from tqdm import tqdm
from pathlib import Path
from args import Args
from loguru import logger

import os
import sys
import torch
import settings

def setup_logger():
    # 设置logger
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}")
    logger.add("./log/log{time}.log", level="INFO", rotation="20 MB")

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
        for batch in tqdm(train_loader, desc=f'epoch {epoch}'):
            # 在加载数据时将数据转移到Device上
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            y_hat = model(x)
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
                x = batch['x'].to(device)
                y = batch['y'].to(device)

                y_hat = model(x)
                avg_NSE += HybirdModel.NSE(y_hat, y)
                avg_RMSE += HybirdModel.RMSE(y_hat, y)
                avg_MAE += HybirdModel.MAE(y_hat, y)
                avg_Bias += HybirdModel.Bias(y_hat, y)
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
        
        data_interface = DataInterface()
        train_loader = data_interface.get_data_loader(opts.train_start_time, opts.train_end_time, opts.batch_size)
        val_loader = data_interface.get_data_loader(opts.val_start_time, opts.val_end_time, opts.batch_size)

        model = HybirdModel(dynamic_input_dim=12, num_timestep=8, lead_time=6)

        # 检查是否要加载预训练的模型
        if opts.pretrain is not None:
            model.load_state_dict(torch.load(opts.pretrain, weights_only=True))
            logger.info(f'Load pretrain model from {opts.pretrain}')

        if opts.use_GPU:
            device = f'cuda:{opts.GPU_id}'
        else:
            device = 'cpu'
        
        optimizer = torch.optim.Adam(model.parameters())
        Loss = torch.nn.MSELoss()
        logger.info('Train on ' + device)
        logger.info('Start training.')
        for epoch in range(opts.start_epoch, opts.epoch + 1):
            self._train_epoch(model, train_loader, optimizer, Loss, epoch, device)

            # 检查是否需要验证
            if opts.val_freq is not None and epoch % opts.val_freq == 0:
                self._validate(model, val_loader, device)

            # 每save_freq保存一次模型, 训练的最后一个epoch也需要保存模型
            if epoch % opts.save_freq == 0 or epoch == opts.epoch:
                self._save_model(model, epoch, opts.checkpoints_dir)

if __name__ == '__main__':
    args = Args()
    args.set_train_args()
    setup_logger()
    train_interface = TrainInterface(args.get_opts())
    train_interface.main()

# python train.py --batch_size=256 --train_start_time=1999-10-01T00 --train_end_time=2004-10-01T00 --epoch=50 --save_freq=5 --use_GPU --GPU_id=0 --val_freq=5 --val_start_time=1996-10-01T00 --val_end_time=1998-10-01T00
# python train.py --batch_size=256 --train_start_date=1999-10-01 --train_end_date=2002-09-30 --epoch=20 --save_freq=5 --use_GPU --GPU_id=0 --val_freq=5 --pretrain=./checkpoints/epoch10.tar --val_start_date=1995-10-01 --val_end_date=1997-09-30 --start_epoch=11

# scp -rP 54212 ./data/CAMELS_US/hourly/ root@connect.yza1.seetacloud.com:/root/autodl-tmp