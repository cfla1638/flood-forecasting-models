import argparse

class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def set_train_args(self):
        self.parser.add_argument("--batch_size", type=int, default=256, help='batch size')
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
        self.parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")
        
        self.parser.add_argument("--train_basin_list", type=str, default=None, help="Basins list for training")
        self.parser.add_argument("--val_basin_list", type=str, default=None, help="Basins list for validating")
        self.parser.add_argument("--train_dataset_path", type=str, default=None, help="Dataset path for training")
        self.parser.add_argument("--val_dataset_path", type=str, default=None, help="Dataset path for validating")
        
        self.parser.add_argument("--dynamic_meanstd", type=str, default=None, help="dynamic meanstd file")
        self.parser.add_argument("--static_meanstd", type=str, default=None, help="stsatic meanstd file")

        self.parser.add_argument("--validate", action="store_true", help="identify whether to validate")
        self.opts = self.parser.parse_args()

    def set_test_args(self):
        self.parser.add_argument("--batch_size", type=int, default=256)
        self.parser.add_argument("--use_GPU", action="store_true", help="identify whether to use gpu")
        self.parser.add_argument("--GPU_id", type=int, default=0, help="device id")
        self.parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")

        self.parser.add_argument("--model_path", type=str, default='./checkpoints/epoch9.pth', help="model path")
        self.parser.add_argument("--basin_list", type=str, default='32_basin_list.txt', help="Basin list for testing")
        self.parser.add_argument("--start_time", type=str, default='2005-10-01T00', help="Testing start time")
        self.parser.add_argument("--end_time", type=str, default='2007-09-30T00', help="Testing end time")
        self.parser.add_argument("--dynamic_meanstd", type=str, default=None, help="dynamic meanstd file")
        self.parser.add_argument("--static_meanstd", type=str, default=None, help="stsatic meanstd file")
        
        # 模式1: 逐个测试单个流域
        self.parser.add_argument("--test_basin_by_basin", action="store_true", default=False, help="Test basin by basin")

        # 模式2: 测试单个流域
        self.parser.add_argument("--test_for_single_basin", action="store_true", default=False, help="Test for single basin")
        self.parser.add_argument("--gauge_id", type=str, help="Gauge id for testing")

        self.opts = self.parser.parse_args()

    def get_opts(self):
        return self.opts
    