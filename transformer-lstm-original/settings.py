from pathlib import Path

basin_list_dir = Path('../data/basin_list/train_val_test/')     # 流域编号列表目录

dataset_path = Path('../data/CAMELS_US/hourly/usgs-streamflow-nldas_hourly.nc')      # 数据集路径

meanstd_dir = Path('../data/meanstd/hourly/')

dynamic_mean_std = None

example_basin = '01013500'  # 供测试使用的流域编号

# 存储DataInterface加载的流域标号
basins_file = 'Region_03_test.txt'

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
target_var = ['qobs_mm_per_hour']    # 目标变量

device = 'cuda:0'