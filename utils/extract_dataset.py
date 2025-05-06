# 根据时间范围和流域，对原始的数据集`.nc`进行分割，将它分割成若干小文件
import numpy as np
import xarray
from pathlib import Path

def load_basin_list(file_path: Path):
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

def extract_dataset(dataset_path, basins):
    start_time = np.datetime64('2002-10-01T00')
    end_time = np.datetime64('2007-09-30T00')
    dataset = xarray.open_dataset(dataset_path)
    dataset = dataset.sel(basin=basins)
    dataset = dataset.sel(date=slice(start_time, end_time))
    dataset.to_netcdf('30_basin_list_evenly_train_5y.nc')

if __name__ == '__main__':
    basin_list_path = Path('../data/basin_list/30_basin_list_evenly.txt')
    dataset_path = Path('../data/CAMELS_US/hourly/usgs-streamflow-nldas_hourly.nc')
    extract_dataset(dataset_path, load_basin_list(basin_list_path)) 
