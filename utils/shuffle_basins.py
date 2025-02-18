import numpy as np
import xarray as xr
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
    
def save_list(list, path):
    with open(path, 'w') as f:
        for item in list:
            f.write(f'{item}\n')

basin_list_path = Path('../data/basin_list/huc_516/Region_03.txt')
basin_list = load_basin_list(basin_list_path)

# 打乱流域列表
np.random.seed(8010)
np.random.shuffle(basin_list)

# 保存20个流域为训练集，10个流域为验证集，10个流域为测试集, 保存到文件中，每行一个流域
train_basin_list = basin_list[:20]
val_basin_list = basin_list[20:30]
test_basin_list = basin_list[30:40]

save_list(train_basin_list, Path('./Region_03_train.txt'))
save_list(val_basin_list, Path('./Region_03_val.txt'))
save_list(test_basin_list, Path('./Region_03_test.txt'))
