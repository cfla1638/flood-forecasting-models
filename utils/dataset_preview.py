# 结论：
# 90%的流域在 1996-10-10 04:00:00 - 2019-03-14 11:00:00 内有数据

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def visualize_variable(dataset, basin, start_date, end_date, variable_name):
    """
    将给定盆地和时间段内的指定变量可视化为折线图。

    参数：
    dataset (xr.Dataset)：xarray 数据集，包含日期和盆地维度及相关变量。
    basin (str)：盆地代码，例如 '08196000'。
    start_date (str 或 datetime)：时间段的起始日期，例如 '1979-01-01'。
    end_date (str 或 datetime)：时间段的结束日期，例如 '1979-12-31'。
    variable_name (str)：要可视化的变量名，例如 'longwave_radiation'。
    """
    # 从数据集中选择指定盆地和时间段的数据，并提取目标变量
    da = dataset.sel(basin=basin, date=slice(start_date, end_date))[variable_name]
    
    # 创建图形和坐标轴，设置图形大小为 10x5
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 绘制变量随时间变化的折线图
    da.plot(ax=ax)
    
    # 设置图表标题，包含变量名和盆地代码
    ax.set_title(f"{variable_name} for Basin: {basin}")
    
    # 设置 Y 轴标签，如果变量有单位则包含单位，否则仅使用变量名
    if 'units' in da.attrs:
        ax.set_ylabel(f"{variable_name} ({da.attrs['units']})")
    else:
        ax.set_ylabel(variable_name)
    
    # 设置 X 轴标签为“Date”
    ax.set_xlabel("Date")
    
    # 显示图表
    plt.show()

def get_dates_with_data(dataset: xr.Dataset, basin: str) -> xr.DataArray:
    """
    返回指定 basin 在 qobs_mm_per_hour 维度上有数据的日期。
    
    参数：
    dataset (xr.Dataset)：xarray 数据集，包含日期和盆地维度及相关变量。
    basin (str)：盆地代码，例如 '08196000'。
    
    返回：
    xr.DataArray：包含有数据的日期的 DataArray。
    """
    # 选择指定 basin 的 qobs_mm_per_hour 数据
    da = dataset.sel(basin=basin)['qobs_mm_per_hour']
    
    # 找到 qobs_mm_per_hour 不为 NaN 的日期
    dates_with_data = da.date[da.notnull()]
    
    return dates_with_data

def get_periods_with_majority_data(dataset: xr.Dataset, variable_name: str = 'qobs_mm_per_hour', threshold_ratio: float = 0.5):
    """
    返回大多数流域在指定变量上有数据的连续时间段。
    
    参数：
    - dataset (xr.Dataset)：包含日期和流域维度及相关变量的数据集。
    - variable_name (str)：要检查的变量名，默认为 'qobs_mm_per_hour'。
    - threshold_ratio (float)：定义“大多数”的比例阈值，默认为 0.5（50%）。
    
    返回：
    - list of tuples：包含连续时间段的列表，每个时间段由 (start_date, end_date) 表示。
    """
    # 选择指定变量
    da = dataset[variable_name]
    
    # 计算每个时间点有多少个流域有数据（非 NaN）
    data_coverage = da.notnull().sum(dim='basin')
    
    # 总流域数量
    total_basins = len(dataset.basin)
    
    # 计算覆盖率（有数据的流域比例）
    coverage_ratio = data_coverage / total_basins
    
    # 筛选出覆盖率超过阈值的时间点
    sufficient_coverage_dates = dataset.date.where(coverage_ratio >= threshold_ratio).dropna(dim='date')
    
    # 将日期转换为 pandas Series 以识别连续时间段
    dates = pd.Series(sufficient_coverage_dates.values)
    
    # 如果没有满足条件的时间点，返回空列表
    if len(dates) == 0:
        return []
    
    # 计算日期差分，识别连续段（假设时间间隔为1小时）
    date_diff = dates.diff().dt.total_seconds() / 3600
    breaks = date_diff > 1  # 间隔大于1小时表示中断
    
    # 分割连续时间段
    periods = []
    start = dates.iloc[0]
    for i in range(1, len(dates)):
        if breaks.iloc[i]:
            end = dates.iloc[i-1]
            periods.append((start, end))
            start = dates.iloc[i]
    periods.append((start, dates.iloc[-1]))  # 添加最后一个时间段
    
    return periods

if __name__ == '__main__':
    dataset_path = Path('../data/CAMELS_US/hourly/usgs-streamflow-nldas_hourly.nc')
    dataset = xr.open_dataset(dataset_path)
    # visualize_variable(dataset, '02472000', '1979-01-01T13', '2019-03-14T12', 'qobs_mm_per_hour')
    # print(get_dates_with_data(dataset, '02469800'))

    periods = get_periods_with_majority_data(dataset, variable_name='qobs_mm_per_hour', threshold_ratio=0.7)
    for start, end in periods:
        print(f"开始时间: {start}, 结束时间: {end}")