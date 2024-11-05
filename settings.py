from pathlib import Path

basin_list_dir = Path('./data/config/')     # 流域编号列表目录

dataset_dir = Path('./data/CAMELS_US')      # 数据集路径

example_basin = '01013500'  # 供测试使用的流域编号

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

device = 'cuda:0'