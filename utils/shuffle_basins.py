# 抽取数据集
import numpy as np
import xarray as xr
from pathlib import Path

def load_basin_list(file_path: Path):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f'{file_path} not found')
    with open(file_path, 'r') as f:
        return [basin.strip() for basin in f.readlines()]
    
def save_list(list, path):
    with open(path, 'w') as f:
        for item in list:
            f.write(f'{item}\n')

def get_basin_list(file_path: Path, cnt: int, seed: int = 42):
    """
    从流域列表中随机抽取流域
    Parameters:
     - file_path: 流域列表文件
     - cnt: 抽取的流域数目
     - seed: 随机种子
    """
    basin_list = load_basin_list(file_path)
    np.random.seed(seed)
    np.random.shuffle(basin_list)
    return basin_list[:cnt]

def get_basin_list_from_clusters(cluster_dir: Path, cnt_per_cluster: int, seed : int = 42):
    """
    从聚类文件中均匀地随机抽取流域
    Parameters:
     - cluster_dir: 聚类文件所在目录
     - cnt_per_cluster: 每个聚类中抽取的流域数目
     - seed: 随机种子
    """
    cluster_files = list(cluster_dir.glob("cluster_*.txt"))
    if not cluster_files:
        raise FileNotFoundError(f"No cluster files found in {cluster_dir}")

    basin_list = []
    for idx, file in enumerate(cluster_files):
        temp_list = load_basin_list(file)
        np.random.seed(seed)
        np.random.shuffle(temp_list)
        basin_list.extend(temp_list[25:50])
    
    return basin_list

def num_cluster_(cluster_dir: Path, tar_basin_list):
    """
    检查某个流域列表中的流域分别属于哪个聚类
    Parameters:
     - cluster_dir: 聚类文件所在目录
     - tar_basin_list: 流域列表文件
    """
    # 读取所有聚类文件
    cluster_files = list(cluster_dir.glob("cluster_*.txt"))
    if not cluster_files:
        raise FileNotFoundError(f"No cluster files found in {cluster_dir}")
    
    for idx, file in enumerate(cluster_files):
        cluster_id = file.stem.split("_")[-1]  # 获取类别编号
        basin_list = load_basin_list(file)

        # 获取相交的流域数目
        intersected_basins = set(tar_basin_list) & set(basin_list)
        print(f"Cluster {cluster_id}: {len(intersected_basins)}")

if __name__ == '__main__':
    cluster_dir = Path("../data/basin_list/clustered_basins/")
    res = get_basin_list_from_clusters(cluster_dir, 5, seed=42)
    num_cluster_(cluster_dir, res)
    save_list(res, Path("../data/basin_list/120_basin_list_evenly.txt"))

# 30_basin_list_evenly.txt             0:5
# 30_basin_list_evenly_test.txt        5:10
# 30_basin_list_evenly_test_1.txt      10:15
# 120_basin_list_evenly.txt            25:50