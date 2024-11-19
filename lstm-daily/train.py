from model import MyModel
from daily_data import DataInterface
from rich.progress import track
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
    logger.add(sys.stdout, level="INFO")
    logger.add("./log/log{time}.log", level="INFO", rotation="20 MB")

class TrainInterface(object):
    def __init__(self, opts) -> None:
        self.opts = opts

    @staticmethod
    def _train_epoch(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch: int, device: str):
        """进行一个epoch的训练, 打印本轮训练的平均损失
        Parameters:
         - model: 要训练的模型
         - train_loader: 加载训练数据, Dataloader的子类
         - optimizer: 优化器
         - epoch: 当前在进行的epoch的编号
         - device: 训练设备, 例如'cpu', 'cuda:0'
        """
        model.train()
        model.to(device)
        optimizer.zero_grad()

        num_batch = 0.0
        losssum = 0.0
        for batch in track(train_loader, description=f'epoch {epoch}'):
            # 在加载数据时将数据转移到Device上
            x_h = batch['x_h'].to(device)
            x_f = batch['x_f'].to(device)
            x_s = batch['x_s'].to(device)
            y = batch['y'].to(device)

            # 将数据转化为(seq_len, batch_size, dim)的格式
            x_h = x_h.transpose(0, 1)
            x_f = x_f.transpose(0, 1)

            y_hat = model(x_s, x_h, x_f)
            loss = model.calculate_loss(y_hat, y)
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
        model.eval()

        num_batch = 0.0
        avg_NSE = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # 在加载数据时将数据转移到Device上
                x_h = batch['x_h'].to(device)
                x_f = batch['x_f'].to(device)
                x_s = batch['x_s'].to(device)
                y = batch['y'].to(device)

                # 将数据转化为(seq_len, batch_size, dim)的格式
                x_h = x_h.transpose(0, 1)
                x_f = x_f.transpose(0, 1)

                y_hat = model(x_s, x_h, x_f)
                y_hat = MyModel.predict(y_hat)
                avg_NSE += MyModel.NSE(y_hat, y)
                num_batch += 1
        logger.info(f'Average NSE: {avg_NSE / num_batch: .4f}')


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
    
    def main(self):
        """训练的流程
        """
        opts = self.opts
        if not os.path.exists(opts.checkpoints_dir):
            os.mkdir(opts.checkpoints_dir)
        
        data_interface = DataInterface()
        train_loader = data_interface.get_data_loader(opts.train_start_date, opts.train_end_date, opts.batch_size)
        val_loader = data_interface.get_data_loader(opts.val_start_date, opts.val_end_date, opts.batch_size)

        model = MyModel(dynamic_input_dim=7,
                static_input_dim=27,
                hidden_dim=256)

        # 检查是否要加载预训练的模型
        if opts.pretrain is not None:
            model.load_state_dict(torch.load(opts.pretrain, weights_only=True))
        
        if opts.use_GPU:
            device = f'cuda:{opts.GPU_id}'
        else:
            device = 'cpu'
        
        optimizer = torch.optim.Adam(model.parameters())
        logger.info('Train on ' + device)
        logger.info('Start training.')
        for epoch in range(opts.start_epoch, opts.epoch + 1):
            self._train_epoch(model, train_loader, optimizer, epoch, device)

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

# python train.py --batch_size=256 --train_start_date=1999-10-01 --train_end_date=2009-09-30 --epoch=50 --save_freq=5 --use_GPU --GPU_id=0 --val_freq=5 --val_start_date=1996-10-01 --val_end_date=1999-09-30
# python train.py --batch_size=256 --train_start_date=1999-10-01 --train_end_date=2002-09-30 --epoch=20 --save_freq=5 --use_GPU --GPU_id=0 --val_freq=5 --pretrain=./checkpoints/epoch10.tar --val_start_date=1995-10-01 --val_end_date=1997-09-30 --start_epoch=11