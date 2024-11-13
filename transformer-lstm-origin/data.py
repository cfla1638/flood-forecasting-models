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

def load_xarray_dataset(dataset_path: Path, basins: List[str], meanstd: pd.DataFrame = None) -> xr.Dataset:
    """
    Parameters:
     - dataset_path: netCDF文件的路径
     - basins: 流域编号的列表
     - meanstd: 存储数据的均值和方差, 用于对数据进行normalization, 
                其index要与df加载的属性相同, 默认为None, 表示利用数据本身的均值方差进行标准化.
                当要用不同流域的数据进行测试时, 需要用训练集的均值方差进行normalization.
    Return:
     - xarray.Dataset: 数据集
    """
    dataset = xr.open_dataset(dataset_path)
    basins = [basin for basin in basins if basin in dataset.coords['basin']]    # 只保留数据集中存在的数据
    print(dataset.sel(basin=basins))

if __name__ == '__main__':
    basin_list_path = settings.basin_list_dir / settings.basins_file
    load_xarray_dataset(settings.dataset_path, load_basin_list(basin_list_path))