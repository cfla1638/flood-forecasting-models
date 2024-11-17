import argparse
import torch

class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def set_train_args(self):
        self.parser.add_argument("--batch_size", type=int, default=1, help='batch size')
        self.parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        self.parser.add_argument("--epoch", type=int, default=60, help="number of end epoch")
        self.parser.add_argument("--start_epoch", type=int, default=1, help="number of start epoch")
        self.parser.add_argument("--use_GPU", action="store_true", help="identify whether to use gpu")
        self.parser.add_argument("--GPU_id", type=int, default=0, help="device id")
        self.parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
        self.parser.add_argument("--save_freq", type=int, default=20, help="save model frequency (per n epoch)")
        self.parser.add_argument("--pretrain", type=str, default=None, help="pretrain model path")
        self.parser.add_argument("--train_start_time", type=str, default='1999-10-01T00', help="Training start time")
        self.parser.add_argument("--train_end_time", type=str, default='2008-10-01T00', help="Training end time")
        self.parser.add_argument("--val_start_time", type=str, default='1995-10-01T00', help="Validating start time")
        self.parser.add_argument("--val_end_time", type=str, default='1999-10-01T00', help="Validating end time")
        self.parser.add_argument("--val_freq", type=int, default=None, help='Validating model frequency (per n epoch)')

        self.opts = self.parser.parse_args()

    def set_test_args(self):
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--use_GPU", action="store_true", help="identify whether to use gpu")
        self.parser.add_argument("--GPU_id", type=int, default=None, help="device id")
        self.parser.add_argument("--weight_path", type=str,
                             default=r"checkpoints\epoch20.tar",
                             help="load path for model weight")
        self.opts = self.parser.parse_args()

    def get_opts(self):
        return self.opts
    