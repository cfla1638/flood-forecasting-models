from model import MyModel
from data import DataInterface
from tqdm import tqdm
from pathlib import Path
from args import Args

import os
import torch
import settings

class TrainInterface(object):
    def __init__(self, opts) -> None:
        self.opts = opts

    @staticmethod
    def _train_epoch(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch: int, device: str):
        model.train()
        model.to(device)
        optimizer.zero_grad()

        num_batch = 0.0
        losssum = 0.0
        for batch in tqdm(train_loader, desc=f'epoch {epoch}'):
            x_h = batch['x_h'].to(device)
            x_f = batch['x_f'].to(device)
            x_s = batch['x_s'].to(device)
            y = batch['y'].to(device)

            x_h = x_h.transpose(0, 1)
            x_f = x_f.transpose(0, 1)

            y_hat = model(x_s, x_h, x_f)
            loss = model.calculate_loss(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            losssum += loss.detach().cpu().numpy().item()
            num_batch += 1
        print(f'Epoch {epoch} | average loss = {losssum / num_batch : .4f}')

    
    @staticmethod
    def _validate(model: torch.nn.Module, val_loader:torch.utils.data.DataLoader, device: str):
        model.eval()

        num_batch = 0.0
        avg_NSE = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_h = batch['x_h'].to(device)
                x_f = batch['x_f'].to(device)
                x_s = batch['x_s'].to(device)
                y = batch['y'].to(device)

                x_h = x_h.transpose(0, 1)
                x_f = x_f.transpose(0, 1)

                y_hat = model(x_s, x_h, x_f)
                y_hat = MyModel.predict(y_hat)
                avg_NSE += MyModel.NSE(y_hat, y)
                num_batch += 1
        print(f'Average NSE: {avg_NSE / num_batch: .4f}')


    @staticmethod
    def _save_model(model, epoch, save_dir: Path):
        save_dir = Path(save_dir)
        model_name = f'epoch{epoch}.tar'
        save_path = save_dir / model_name
        torch.save(model, save_path)
    
    def main(self):
        opts = self.opts
        if not os.path.exists(opts.checkpoints_dir):
            os.mkdir(opts.checkpoints_dir)
        
        data_interface = DataInterface()
        train_loader = data_interface.get_data_loader(opts.train_start_date, opts.train_end_date, opts.batch_size)
        val_loader = data_interface.get_data_loader(opts.val_start_date, opts.val_end_date, opts.batch_size)

        if opts.pretrain is None:
            model = MyModel(dynamic_input_dim=7,
                static_input_dim=27,
                hidden_dim=256)
        else:
            model = torch.load(opts.pretrain)
        
        if opts.use_GPU:
            device = f'cuda:{opts.GPU_id}'
        else:
            device = 'cpu'
        
        optimizer = torch.optim.Adam(model.parameters())
        print(' - Train on ' + device)
        print(' - Start training.')
        for epoch in range(opts.start_epoch, opts.epoch + 1):
            self._train_epoch(model, train_loader, optimizer, epoch, device)
            if opts.val_freq is not None and epoch % opts.val_freq == 0:
                self._validate(model, val_loader, device)

            if epoch % opts.save_freq == 0 or epoch == opts.epoch:
                self._save_model(model, epoch, opts.checkpoints_dir)

if __name__ == '__main__':
    args = Args()
    args.set_train_args()
    train_interface = TrainInterface(args.get_opts())
    train_interface.main()

# python train.py --batch_size=256 --train_start_date=1999-10-01 --train_end_date=2002-09-30 --epoch=10 --save_freq=5 --use_GPU --GPU_id=0
# python train.py --batch_size=256 --train_start_date=1999-10-01 --train_end_date=2002-09-30 --epoch=20 --save_freq=5 --use_GPU --GPU_id=0 --val_freq=5 --pretrain=./checkpoints/epoch10.tar --val_start_date=1995-10-01 --val_end_date=1997-09-30 --start_epoch=11