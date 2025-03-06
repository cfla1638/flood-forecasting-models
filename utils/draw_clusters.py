# Description: 绘制聚类后的流域地图
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

def load_basin_list(file_path: Path):
    """读取流域列表"""
    if not file_path.is_file():
        raise FileNotFoundError(f'{file_path} not found')
    with open(file_path, 'r') as f:
        return [basin.strip() for basin in f.readlines()]

def draw_clusters(output_dir: Path, dataset_path: Path):
    """绘制聚类后的流域地图
    Parameters:
    - output_dir: 存放聚类结果的目录
    - dataset_path: CAMELS_US 数据集目录（包含 gauge_information.txt）
    """
    # 读取流域元数据
    basin_metadata = pd.read_csv(dataset_path / 'basin_metadata/gauge_information.txt',
                                 index_col='GAGE_ID', sep='\t', engine='python',
                                 dtype={'GAGE_ID': str, 'HUC_02': str})

    # 读取所有聚类文件
    cluster_files = list(output_dir.glob("cluster_*.txt"))
    if not cluster_files:
        raise FileNotFoundError(f"No cluster files found in {output_dir}")

    # 定义颜色映射
    colors = plt.cm.get_cmap("tab10", len(cluster_files))

    # 创建绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, file in enumerate(cluster_files):
        cluster_id = file.stem.split("_")[-1]  # 获取类别编号
        basin_list = load_basin_list(file)

        # 筛选位于该类别中的流域站点
        cluster_basins = basin_metadata.loc[basin_list]

        # 转换为 GeoDataFrame
        gdf = gpd.GeoDataFrame(cluster_basins, 
                               geometry=gpd.points_from_xy(cluster_basins["LONG"], cluster_basins["LAT"]),
                               crs="EPSG:4326")

        # 重新投影到 Web Mercator (EPSG:3857) 以适配底图
        gdf = gdf.to_crs(epsg=3857)

        # 绘制站点
        gdf.plot(ax=ax, color=colors(idx), markersize=50, alpha=0.8, edgecolor="black", label=f"Cluster {cluster_id}")

    # 添加底图
    tile_url = "http://wprd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=7&x={x}&y={y}&z={z}"
    ctx.add_basemap(ax, source=tile_url, alpha=0.5)

    # 设置图例和标题
    ax.legend()
    ax.set_title("不同类别流域的分布", fontsize=14)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")

    # 移除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 显示地图
    plt.show()

# 示例调用
output_dir = Path("../data/basin_list/clustered_basins")  # 你的聚类结果目录
dataset_path = Path("../data/CAMELS_US/")  # 你的 CAMELS 数据集目录
draw_clusters(output_dir, dataset_path)
