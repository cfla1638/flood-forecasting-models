import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

dataset_path = './data/CAMELS_US/hourly/30_basin_list_evenly_test_test_2y.nc'
ds = xr.open_dataset(dataset_path)

# 使用动态属性的列表
forcing_attrs = ['convective_fraction',
                  'longwave_radiation',
                  'potential_energy',
                  'potential_evaporation',
                  'pressure',
                  'shortwave_radiation',
                  'specific_humidity',
                  'temperature',
                  'total_precipitation',
                  'wind_u',
                  'wind_v',
                  'qobs_mm_per_hour'
                  ]

# 展示所有的动态属性
for attr in forcing_attrs:
    print(attr)
    print(ds[attr].values)
    print(ds[attr].values.shape)
    print('')