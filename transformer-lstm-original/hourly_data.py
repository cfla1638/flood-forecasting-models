# 数据处理
# 数据 coordinates:
# date: 1979-01-01T13 - 2019-03-14T12
# basin: 516个
import numpy as np
import pandas as pd
import xarray as xr
import sys
import torch
import settings

from pathlib import Path
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from loguru import logger

def load_basin_list(file_path: Path) -> List[str]:
    """读取流域列表
    Parameter:
     - file_path: 文件路径
    Return:
     - 流域编号的列表
    """
    if not file_path.is_file():
        raise FileNotFoundError(f'{file_path} not found')
    with open(file_path, 'r') as f:
        return [basin.strip() for basin in f.readlines()]

def load_xarray_dataset(dataset_path: Path, basins: List[str], 
                        basins_filename: str,
                        mean_std_dir: Path,
                        meanstd: pd.DataFrame = None) -> xr.Dataset:
    """
    Parameters:
     - dataset_path: netCDF文件的路径
     - basins: 流域编号的列表
     - basins_filename: 本次所用basin_list的文件名, 用于命名保存的meanstd文件
     - mean_std_dir: 存储meanstd文件的文件夹
     - meanstd: 存储数据的均值和方差, 用于对数据进行normalization, 
                其index要与df加载的属性相同, 默认为None, 表示利用数据本身的均值方差进行标准化.
                当要用不同流域的数据进行测试时, 需要用训练集的均值方差进行normalization.
    Return:
     - xarray.Dataset: 数据集
    """
    logger.info('Loading dynamic data...')
    dataset = xr.open_dataset(dataset_path)
    logger.info('Dynamic data loaded.')

    # 取给定basin和时间段的数据
    basins = [basin for basin in basins if basin in dataset.coords['basin']]    # 只保留数据集中存在的数据
    dataset = dataset.sel(basin=basins)
    logger.info(f'Loaded dynamic data for {len(basins)} basins.')

    logger.info('processing dynamic data...')

    # 处理缺失值
    dataset = dataset.fillna(dataset.mean())

    # Normalization
    if meanstd is None:
        mean = dataset.mean()
        std = dataset.std()
        dataset = (dataset - mean) / std
        
        # 保存均值方差
        m = mean.to_pandas().rename('mean')
        s = std.to_pandas().rename('std')
        ms = pd.concat([m, s], axis=1)
        basins_filename = basins_filename.split('.')[0]
        ms.to_csv(mean_std_dir / ('dynamic_hourly_meanstd_' + basins_filename + '.csv'))
        logger.info(f'Dynamic mean and std saved to {mean_std_dir / ("dynamic_" + basins_filename + ".csv")}')
    else:
        mean = xr.Dataset({var:([], value) for var, value in meanstd['mean'].items()})
        std = xr.Dataset({var:([], value) for var, value in meanstd['std'].items()})
        logger.info('Using provided mean and std for normalization')

    # 将标准差中的 0 替换为一个很小的值，以避免除零错误
    std = std.where(std != 0, other=1e-8)

    dataset = (dataset - mean) / std
    dataset.attrs['mean'] = mean
    dataset.attrs['std'] = std

    logger.info('Dynamic data processed.')

    return dataset

class MyDataset(Dataset):
    def __init__(self, 
                 dynamic_ds: xr.Dataset,        # forcing data
                 start_time: np.datetime64,     
                 end_time: np.datetime64,
                 forcing_attrs: List,           # 用于训练的属性列表
                 target_vars: List,             # 目标变量的列表
                 num_timestep: int = 8,
                 lead_time: int = 6
                 ):
        super().__init__()
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)

        self.dynamic_ds = dynamic_ds
        self.start_time = start_time
        self.end_time = end_time
        self.num_timestep = num_timestep
        self.lead_time = lead_time
        self.forcing_attrs = forcing_attrs
        self.target_vars = target_vars

        self.num_basins = dynamic_ds.dims['basin']  # basin的数目
        num_hours = (end_time - start_time).astype(np.int32)    # 给定时间段共有多少小时
        self.slide_wnd = num_timestep + lead_time   # 滑动窗口长度
        self.num_samples_per_basin = (num_hours - self.slide_wnd + 1)   # 给定时间段共可以生成多少样本(滑动窗口滑动次数)

    
    def __len__(self):
        return self.num_basins * self.num_samples_per_basin

    def __getitem__(self, idx: int):
        # 获取流域id和时间偏移
        sample_idx = int(idx % self.num_samples_per_basin)
        basin_idx = int(idx // self.num_samples_per_basin)
        basin = self.dynamic_ds.coords['basin'][basin_idx].values.item()

        # 计算本样本的开始时间和结束时间
        start_time = self.start_time + sample_idx
        end_time = start_time + self.slide_wnd - 1

        # 筛选本样本的具体数据        
        sample = self.dynamic_ds.sel(date=slice(start_time, end_time))
        sample = sample.isel(basin=basin_idx)

        # 切分数据
        x = sample.isel(date=slice(0, self.num_timestep))
        y = sample.isel(date=slice(-self.lead_time, None))

        x = x[self.forcing_attrs]
        y = y[self.target_vars]

        # 转换成torch.Tensor
        x = x.to_array(dim='variable').transpose('date', 'variable')
        x = torch.from_numpy(x.values).float()

        y = y.to_array(dim='variable').transpose('date', 'variable')
        y = torch.from_numpy(y.values.squeeze()).float()

        return {'x': x, 'y': y}
    
class DataInterface(object):
    def __init__(self, 
                 basins_file : str = None, 
                 default_start_time : str = None, 
                 default_end_time : str = None, 
                 default_numstep : int = 8, 
                 default_lead_time : int = 6,
                 default_batch_size : int = 256,
                 default_num_workers : int = 8
                 ) -> None:
        """
        Parameters:
         - basins_file: 流域列表的文件名, 如果不传入该参数, 则使用settings.basins_file
         - default_start_time: 默认的开始时间, 如果仅仅使用get_data_loader方法, 可以不传入该参数
         - default_end_time: 默认的结束时间, 如果仅仅使用get_data_loader方法, 可以不传入该参数
         - default_numstep: 默认的时间步长, 如果仅仅使用get_data_loader方法, 可以不传入该参数
         - default_lead_time: 默认的预测时间步长, 如果仅仅使用get_data_loader方法, 可以不传入该参数
         - default_batch_size: 默认的batch_size, 如果仅仅使用get_data_loader方法, 可以不传入该参数
         - default_num_workers: 默认的num_workers, 如果仅仅使用get_data_loader方法, 可以不传
        """
        # 处理默认参数
        if basins_file is None:
            basins_file = settings.basins_file
        logger.info(f'Using basin list: {basins_file}')
        self.default_start_time = default_start_time
        self.default_end_time = default_end_time
        self.default_numstep = default_numstep
        self.default_lead_time = default_lead_time
        self.default_batch_size = default_batch_size
        self.default_num_workers = default_num_workers

        # 加载标准化数据的均值方差
        dynamic_meanstd = None
        if settings.dynamic_mean_std is not None:
            dynamic_meanstd = pd.read_csv(settings.mean_std_dir / settings.dynamic_mean_std, index_col=0)

        # 加载数据
        self.basin_list = load_basin_list(settings.basin_list_dir / basins_file)
        self.dataset = load_xarray_dataset(settings.dataset_path, self.basin_list, 
                                           settings.basins_file, settings.meanstd_dir, dynamic_meanstd)

    def get_data_loader(self, start_time: str = None, end_time: str = None, batch_size: int = 256,
                        num_timestep: int = 8, lead_time: int = 6, num_workers: int = 8):
        if start_time is None:
            start_time = self.default_start_time
        if end_time is None:
            end_time = self.default_end_time

        dataset = MyDataset(self.dataset, start_time, end_time,
                            settings.forcing_attrs, settings.target_var, num_timestep, lead_time)
        return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
    
    def __len__(self):
        return len(self.basin_list)
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.basin_list):
            raise IndexError(f'Index {idx} out of range')
        this_basin = self.dataset.sel(basin=self.basin_list[idx]).expand_dims('basin')  # 选择一个流域后, basin维度会消失, 因此需要用expand_dims重新加上
        dataset = MyDataset(this_basin, self.default_start_time, self.default_end_time,
                            settings.forcing_attrs, settings.target_var, self.default_numstep, self.default_lead_time)
        loader = DataLoader(dataset, self.default_batch_size, shuffle=True, num_workers=self.default_num_workers)
        return self.basin_list[idx], loader


if __name__ == '__main__':
    logger.remove() # 禁用日志

    datahub = DataInterface('Region_03_train.txt', '1990-01-01T00', '1995-01-01T00')
    for basin, loader in datahub:
        print(basin)
        input()
    # loader = datahub.get_data_loader('1990-01-01T00', '1995-01-01T00')
    # for batch in loader:
    #     print(batch['x'].shape)
    #     print(batch['y'].shape)
    #     input()