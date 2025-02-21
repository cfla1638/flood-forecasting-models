from pathlib import Path

basin_list_dir = Path('../data/basin_list/train_val_test/')     # 流域编号列表目录

dataset_dir = Path('../data/CAMELS_US')      # 数据集路径

dataset_path = Path('../data/CAMELS_US/hourly/usgs-streamflow-nldas_hourly.nc')      # 数据集路径

meanstd_dir = Path('../data/meanstd/hourly/')

dynamic_mean_std = None   # 用于标准化静态数据的均值方差, 可以为None
static_mean_std = None    # 用于标准化静态数据的均值方差, 可以为None

example_basin = '01013500'  # 供测试使用的流域编号

# 存储DataInterface加载的流域标号
basins_file = 'Region_03_train.txt'

# 使用静态属性的列表
attribute_list = ['p_mean',
                  'pet_mean',
                  'aridity',
                  'p_seasonality',
                  'frac_snow',
                  'high_prec_freq',
                  'high_prec_dur',
                  'low_prec_freq',
                  'low_prec_dur',
                  'elev_mean',
                  'slope_mean',
                  'area_gages2',
                  'frac_forest',
                  'lai_max',
                  'lai_diff',
                  'gvf_max',
                  'gvf_diff',
                  'soil_depth_pelletier',
                  'soil_depth_statsgo',
                  'soil_porosity',
                  'soil_conductivity',
                  'max_water_content',
                  'sand_frac',
                  'silt_frac',
                  'clay_frac',
                  'carbonate_rocks_frac',
                  'geol_porostiy'
                  ]

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