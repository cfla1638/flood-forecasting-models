import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import contextily as ctx

# 读取数据
dataset_path = Path('../data/CAMELS_US/')      # 数据集路径

basin_metadata = pd.read_csv(dataset_path / 'basin_metadata/gauge_information.txt', sep='\t', engine='python', dtype={'GAGE_ID': str, 'HUC_02': str})

# 展示不同 HUC_02 中的流域数量
# print(basin_metadata['HUC_02'].value_counts())

# HUC_02
# 03    92
# 17    91
# 02    75
# 10    70
# 05    45
# 18    40
# 12    37
# 07    33
# 11    31
# 04    31
# 01    27
# 15    19
# 16    18
# 14    17
# 06    17
# 08    12
# 09     9
# 13     7
# Name: count, dtype: int64

