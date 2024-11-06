import numpy as np
import pandas as pd
import xarray as xr
import sys
import torch

from pathlib import Path
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import settings

def load_camels_us_forcings(data_dir: Path, basin: str, forcings: str) -> Tuple[pd.DataFrame, int]:
    """加载CAMELS US 数据集的 forcing data
    Parameters:
     - data_dir: CAMELS_US的路径
     - basin: 流域的编号
     - forcings: forcings的数据源,可以是'daymet', 'maurer', 'nldas'
    
    Returns:
     - DataFrame: index为时间
     - int: 流域面积

    示例: df, area =  load_camels_us_forcings(Path('./data/CAMELS_US'), '01013500', 'daymet')
    """
    forcing_path = data_dir / 'basin_mean_forcing' / forcings # 文件夹下为18个编号从1-18的文件夹，存放着不同地区的forcing data
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    file_path = list(forcing_path.glob(f'**/{basin}_*_forcing_leap.txt'))
    if file_path:
        file_path = file_path[0]    # 由于只可能找到唯一的forcing data文件，因此取列表的第一个项目
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')
    
    with open(file_path, 'r') as fp:
        # load area from header
        fp.readline()   # 忽略 latitude of gauge
        fp.readline()   # 忽略 elevation of gauge (m)
        area = int(fp.readline())   # 读取面积(m^2)

        df = pd.read_csv(fp, sep='\s+')
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
                                    format="%Y/%m/%d")
        df = df.set_index("date")
    return df, area

def load_camels_us_discharge(data_dir: Path, basin: str, area: int) -> pd.Series:
    """加载 CAMELS US 数据集的流量
    Parameters:
     - data_dir: CAMELS_US的路径
     - basin: 流域的编号
     - area: 流域面积，用于对流量进行标准化
    """
    discharge_path = data_dir / 'usgs_streamflow'
    file_path = list(discharge_path.glob(f'**/{basin}_streamflow_qc.txt'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
    df = df.set_index("date")

    # normalize discharge from cubic feet per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs

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

def load_basin_data(data_dir: Path, basin: str, forcing: str) -> pd.DataFrame:
    """加载组合forcing data和discharge的数据,并将discharge中所有的-999替换为nan
    Parameters:
     - data_dir: CAMELS_US的路径
     - basin: 流域的编号
     - forcings: forcings的数据源,可以是'daymet', 'maurer', 'nldas'
    """
    df, area = load_camels_us_forcings(data_dir, basin, forcing)
    df['QObs(mm/d)'] = load_camels_us_discharge(data_dir, basin, area)
    df.loc[df['QObs(mm/d)'] < 0, 'QObs(mm/d)'] = np.nan
    return df

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
    
def normalization(dataset: xr.Dataset) -> xr.Dataset:
    """标准化
    对一个Dataset中所有流域的所有forcing data进行标准化, 减去均值除以方差, 
    并将均值方差存储在 ``dataset.attrs['mean']`` 和 ``dataset.attrs['std']``
    """
    mean = dataset.mean()
    std = dataset.std()
    
    # 将标准差中的 0 替换为一个很小的值，以避免除零错误
    std = std.where(std != 0, other=1e-8)

    dataset = (dataset - mean) / std
    dataset.attrs['mean'] = mean
    dataset.attrs['std'] = std
    return dataset

def load_xarray_dataset(data_dir: Path, basins: List[str]) -> xr.Dataset:
    """构造xarray数据集, 并进行标准化
    Parameters:
     - data_dir: CAMELS_US的路径
     - basins: 流域编号的列表
    Return:
     - xarray.Dataset: 数据集
    """
    # 加载并拼接dataset
    data_list = []
    for basin in tqdm(basins, file=sys.stdout, desc='Loading basin data'):
        df = load_basin_data(data_dir, basin, 'daymet')
        ds = xr.Dataset.from_dataframe(df.astype(np.float32))
        ds = ds.assign_coords({'basin': basin})
        data_list.append(ds)
    dataset = xr.concat(data_list, dim='basin')

    # normalization
    dataset = normalization(dataset)
   
    return dataset

def load_static_attributes(df: pd.DataFrame, attrs: List[str]):
    df = df[attrs]      # 取出指定的属性列
    df = df.fillna(df.mean())   # 填充缺失值

    mean = df.mean()    
    std = df.std()
    
    # 如果只有一行数据，std 会全部为 0，我们可以直接将 std 设置为 1 来避免除零
    if len(df) == 1:
        std = std.fillna(1.0)
    else:
        std = std.replace(0, 1e-8)

    df = (df - mean) / std  # normalization

    return {index : torch.from_numpy(row.values.flatten().astype(np.float32)) for index, row in df.iterrows()}

class MyDataset(Dataset):
    def __init__(self, dynamic_ds: xr.Dataset, 
                 static_ds: Dict[str, torch.Tensor],
                 start_date: np.datetime64,      # 开始时间，包括该天
                 end_date: np.datetime64,        # 结束时间，包括该天
                 hindcast_length: int = 365, 
                 forecast_horizon: int = 7
                 ):
        super().__init__()
        start_date = np.datetime64(start_date)
        end_date = np.datetime64(end_date)

        if start_date > end_date:
            raise Exception('开始时间不应在结束时间之后!')
        if (end_date - start_date).astype(np.int32) + 1 < hindcast_length + forecast_horizon:
            raise Exception('取样时间范围不能小于滑动窗口长度')
        if (dynamic_ds.coords['date'][0].values > start_date) or (dynamic_ds.coords['date'][-1] < end_date):
            raise Exception('取样时间须在数据集的时间范围内')
        self.dynamic_ds = dynamic_ds
        self.static_ds = static_ds
        self.start_date = start_date
        self.end_date = end_date
        self.hindcast_length = hindcast_length
        self.forecast_horizon = forecast_horizon

        self.num_basins = self.dynamic_ds.dims['basin']     # 数据集中流域的总数目
        num_days = (self.end_date - self.start_date).astype(np.int32) + 1   # 数据集中的总天数
        self.slide_wnd = self.hindcast_length + self.forecast_horizon   # 根据后顾时间长度和预测时间长度计算滑动窗口长度
        self.num_samples_per_basin = (num_days - self.slide_wnd + 1)    # 计算数据集中包含的天数可以切分出多少训练数据
    
    def __len__(self):
        return self.num_basins * self.num_samples_per_basin

    def __getitem__(self, idx: int):
        # 获取流域id和时间偏移
        sample_idx = int(idx % self.num_samples_per_basin)
        basin_idx = int(idx / self.num_samples_per_basin)
        basin = self.dynamic_ds.coords['basin'][basin_idx].values.item()
        
        # 计算本样本的开始时间和结束时间
        start_date = self.start_date + sample_idx
        end_date = start_date + self.slide_wnd - 1

        # 筛选本样本的具体数据        
        sample = self.dynamic_ds.sel(date=slice(start_date, end_date))
        sample = sample.isel(basin=basin_idx)
        
        # 切分成后顾数据和预测数据
        hindcast_sample = sample.isel(date=slice(0, self.hindcast_length))
        forecast_sample = sample.isel(date=slice(-self.forecast_horizon, None))
        
        # 将x, y从数据集中取出
        x_h = hindcast_sample[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']]
        x_f = forecast_sample[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']]
        y = forecast_sample['QObs(mm/d)']

        # 转换成torch.Tensor
        x_h = x_h.to_array(dim='variable').transpose('date', 'variable')
        x_h = torch.from_numpy(x_h.values).float()
        
        x_f = x_f.to_array(dim='variable').transpose('date', 'variable')
        x_f = torch.from_numpy(x_f.values).float()

        y = torch.from_numpy(y.values).float()

        x_s = self.static_ds[basin]

        return {'x_h': x_h, 'x_f': x_f, 'x_s': x_s, 'y': y}

class DataInterface(object):
    def __init__(self) -> None:
        # 加载数据
        self.basins = load_basin_list(settings.basin_list_dir / '32_basin_list.txt')
        self.dynamic_ds = load_xarray_dataset(settings.dataset_dir, self.basins)
        self.attrs = load_camels_us_attributes(settings.dataset_dir, self.basins)
        self.static_ds = load_static_attributes(self.attrs, settings.attribute_list)

    def get_data_loader(self, start_date: str, end_date: str, batch_size: int = 256):
        dataset = MyDataset(self.dynamic_ds, self.static_ds, start_date, end_date)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    basins = load_basin_list(settings.basin_list_dir / '10_basin_list.txt')
    dataset = load_xarray_dataset(settings.dataset_dir, basins)
    attrs = load_camels_us_attributes(settings.dataset_dir, basins)
    static_ds = load_static_attributes(attrs, settings.attribute_list)
    ds = MyDataset(dataset, static_ds, '1989-06-04', '1996-06-04')
    loader = DataLoader(ds, 32, True)

    print(len(ds))
    cnt = 0
    for i in loader:
        print(i['x_s'].shape)
        print(i['x_h'].shape)
        input()    