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

# --- 处理静态数据 ---

def load_camels_us_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """读取流域的静态属性
    Parameters:
     - data_dir: CAMELS_US的路径
     - basins: 要加载的流域编号列表，默认返回所有的流域

    Return:
     - pd.DataFrame: index为gauge_id
    """
    attributes_path = data_dir / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')
        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    # convert huc column to double digit strings, 例如，1会被转换为'01'
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df

def load_static_attributes(df: pd.DataFrame, attrs: List[str], basin_filename: str, meanstd_dir: Path, 
                           meanstd: pd.DataFrame = None):
    """加载basin静态属性
    Parameters:
     - df: 以basin编号为index的静态属性Dataframe
     - attrs: 静态属性名的列表
     - basin_filename: 使用的, 存储basin list的文件名, 例如'10_basin_list.txt', 用于命名均值方差文件
     - meanstd_dir: 存储方差文件的文件夹路径
     - meanstd: 以静态属性名为index, 有mean和std两列的Dataframe, 用于normalization
    Return:
     - {basin : torch.Tensor}: 以basin编号为键, 以对应的静态属性tensor为值得列表
    """
    df = df[attrs]      # 取出指定的属性列
    df = df.fillna(df.mean())   # 填充缺失值

    if meanstd is None:
        meanstd_dir = Path(meanstd_dir)
        mean = df.mean().rename('mean')
        std = df.std().rename('std')
        ms = pd.concat([mean, std], axis=1)
        basin_filename = basin_filename.split('.')[0]
        ms.to_csv(meanstd_dir / ('static_meanstd_' + basin_filename + '.csv'))
    else:
        mean = meanstd['mean']
        std = meanstd['std']

    # 如果只有一行数据，std 会全部为 0，我们可以直接将 std 设置为 1 来避免除零
    if len(df) == 1:
        std = std.fillna(1.0)
    else:
        std = std.replace(0, 1e-8)

    df = (df - mean) / std  # normalization

    return {index : torch.from_numpy(row.values.flatten().astype(np.float32)) for index, row in df.iterrows()}


# --- 处理动态数据 ---

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
    dataset = xr.open_dataset(dataset_path)
    
    # 取给定basin和时间段的数据
    basins = [basin for basin in basins if basin in dataset.coords['basin']]    # 只保留数据集中存在的数据
    dataset = dataset.sel(basin=basins)

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
    def __init__(self, 
                 dynamic_ds: xr.Dataset,        # forcing data
                 static_ds: Dict[str, torch.Tensor],    # static data
                 start_time: np.datetime64,     
                 end_time: np.datetime64,
                 forcing_attrs: List,           # 用于训练的属性列表
                 target_vars: List,             # 目标变量的列表
                 hindcast_length: int = 336,    # 336h = 14d
                 forecast_horizon: int = 6
                 ):
        super().__init__()
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)

        self.dynamic_ds = dynamic_ds
        self.static_ds = static_ds
        self.start_time = start_time
        self.end_time = end_time
        self.hindcast_length = hindcast_length
        self.forecast_horizon = forecast_horizon
        self.forcing_attrs = forcing_attrs
        self.target_vars = target_vars

        self.num_basins = dynamic_ds.dims['basin']  # basin的数目
        num_hours = (end_time - start_time).astype(np.int32)    # 给定时间段共有多少小时
        self.slide_wnd = hindcast_length + forecast_horizon   # 滑动窗口长度
        self.num_samples_per_basin = (num_hours - self.slide_wnd + 1)   # 给定时间段共可以生成多少样本(滑动窗口滑动次数)

    
    def __len__(self):
        return self.num_basins * self.num_samples_per_basin

    def __getitem__(self, idx: int):
        # 获取流域id和时间偏移
        sample_idx = int(idx % self.num_samples_per_basin)
        basin_idx = int(idx / self.num_samples_per_basin)
        basin = self.dynamic_ds.coords['basin'][basin_idx].values.item()

        # 计算本样本的开始时间和结束时间
        start_time = self.start_time + sample_idx
        end_time = start_time + self.slide_wnd - 1

        # 筛选本样本的具体数据        
        sample = self.dynamic_ds.sel(date=slice(start_time, end_time))
        sample = sample.isel(basin=basin_idx)

        # 切分数据
        hindcast_sample = sample.isel(date=slice(0, self.hindcast_length))
        forecast_sample = sample.isel(date=slice(-self.forecast_horizon, None))

        x_h = hindcast_sample[self.forcing_attrs]
        x_f = forecast_sample[self.forcing_attrs]
        x_s = self.static_ds[basin]
        y = forecast_sample[self.target_vars]

        # 转换成torch.Tensor
        x_h = x_h.to_array(dim='variable').transpose('date', 'variable')
        x_h = torch.from_numpy(x_h.values).float()

        x_f = x_f.to_array(dim='variable').transpose('date', 'variable')
        x_f = torch.from_numpy(x_f.values).float()

        y = y.to_array(dim='variable').transpose('date', 'variable')
        y = torch.from_numpy(y.values.squeeze()).float()

        return {'x_h': x_h, 'x_f': x_f, 'x_s': x_s, 'y': y}
    
class DataInterface(object):
    def __init__(self) -> None:
        # 加载标准化数据的均值方差
        dynamic_meanstd = None
        static_meanstd = None
        if settings.dynamic_mean_std is not None:
            dynamic_meanstd = pd.read_csv(settings.meanstd_dir / settings.dynamic_mean_std, index_col=0)
        if settings.static_mean_std is not None:
            static_meanstd = pd.read_csv(settings.meanstd_dir / settings.static_mean_std, index_col=0)

        # 加载数据
        basin_list = load_basin_list(settings.basin_list_dir / settings.basins_file)
        self.dynamic_ds = load_xarray_dataset(settings.dataset_path, basin_list, 
                                           settings.basins_file, settings.meanstd_dir, dynamic_meanstd)
        attrs = load_camels_us_attributes(settings.dataset_dir, basin_list)
        self.static_ds = load_static_attributes(attrs, settings.attribute_list, settings.basins_file, settings.meanstd_dir)

    def get_data_loader(self, start_time: str, end_time: str, batch_size: int = 256,
                        hindcast_length: int = 336, forecast_horizon: int = 6, num_workers: int = 8):
        dataset = MyDataset(self.dynamic_ds, self.static_ds, start_time, end_time,
                            settings.forcing_attrs, settings.target_var, hindcast_length, forecast_horizon)
        return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)


if __name__ == '__main__':
    datahub = DataInterface()
    loader = datahub.get_data_loader('1990-01-01T00', '1995-01-01T00', num_workers=1)
    for batch in loader:
        print(batch['x_h'].shape)
        print(batch['x_f'].shape)
        print(batch['x_s'].shape)
        print(batch['y'].shape)
        input()