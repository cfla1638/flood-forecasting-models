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
    

basin_list_path = Path('./data/basin_list/')
save_path = basin_list_path / 'huc_516'
basins_516 = Path('./data/basin_list/516_basins_hourly.txt')
basin_list = load_basin_list(basins_516)

regions = list((basin_list_path / 'huc_671').glob('Region_*.txt'))

# 筛选属于516个流域的流域
for region in regions:
    region_basins = load_basin_list(region)
    for basin in region_basins:
        if basin not in basin_list:
            print(f'region {region.name} has basin {basin} not in 516 basins')
    