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
                        start_time: str, end_time :str, basins_filename: str,
                        mean_std_dir: Path,
                        meanstd: pd.DataFrame = None) -> xr.Dataset:
    """
    Parameters:
     - dataset_path: netCDF文件的路径
     - basins: 流域编号的列表
     - start_time: 开始时间(例如: '1989-06-04T04')
     - end_time: 结束时间
     - basins_filename: 本次所用basin_list的文件名, 用于命名保存的meanstd文件
     - mean_std_dir: 存储meanstd文件的文件夹
     - meanstd: 存储数据的均值和方差, 用于对数据进行normalization, 
                其index要与df加载的属性相同, 默认为None, 表示利用数据本身的均值方差进行标准化.
                当要用不同流域的数据进行测试时, 需要用训练集的均值方差进行normalization.
    Return:
     - xarray.Dataset: 数据集
    """
    dataset = xr.open_dataset(dataset_path)
    
    # 取给定basin和时间段的数据
    basins = [basin for basin in basins if basin in dataset.coords['basin']]    # 只保留数据集中存在的数据
    dataset = dataset.sel(basin=basins, date=slice(start_time, end_time))

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
    else:
        mean = xr.Dataset({var:([], value) for var, value in meanstd['mean'].items()})
        std = xr.Dataset({var:([], value) for var, value in meanstd['std'].items()})

    # 将标准差中的 0 替换为一个很小的值，以避免除零错误
    std = std.where(std != 0, other=1e-8)

    dataset = (dataset - mean) / std
    dataset.attrs['mean'] = mean
    dataset.attrs['std'] = std
    return dataset

class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass

if __name__ == '__main__':
    basin_list = load_basin_list(settings.basin_list_dir / settings.basins_file)
    dataset = load_xarray_dataset(settings.dataset_path, basin_list, 
                        '1990-01-01T00', '1999-01-01T12', 
                        settings.basins_file, settings.meanstd_dir)
    
    print(dataset)