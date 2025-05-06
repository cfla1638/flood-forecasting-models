# 根据不同流域的评价指标，绘制每个流域的表现，并将其绘制在地图上
import numpy as np
import pandas as pd
import re
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import contextily as ctx
from matplotlib.lines import Line2D  # 用于自定义图例

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rc("font", size=12)  # 设置字体大小

def load_basin_list(file_path: Path):
    """读取流域列表
    Parameter:
     - file_path: 文件路径
    Return:
     - 流域编号的列表
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f'{file_path} not found')
    with open(file_path, 'r') as f:
        return [basin.strip() for basin in f.readlines()]

def load_log(filename):
    """
    Load log file and extract metrics.
    """
    with open(filename, "r", encoding="utf-8") as file:
        log_data = file.read()

    pattern = r"Average NSE:\s+([-+]?\d*\.\d+)\s+\|\s+Average RMSE:\s+([-+]?\d*\.\d+)\s+\|\s+Average MAE:\s+([-+]?\d*\.\d+)\s+\|\s+Average Bias:\s+([-+]?\d*\.\d+)"
    
    matches = re.findall(pattern, log_data)
    metrics = [list(map(float, match)) for match in matches]
    return np.array(metrics).transpose()

def load_clusters(cluster_dir = '../data/basin_list/clustered_basins'):
    """
    Load clustered basins from files in the specified directory.
    """
    cluster_files = list(Path(cluster_dir).glob("cluster_*.txt"))
    if not cluster_files:
        raise FileNotFoundError(f"No cluster files found in {cluster_dir}")

    clusters = []
    for file in cluster_files:
        cluster_id = file.stem.split("_")[-1]  # 获取类别编号
        basin_list = load_basin_list(file)
        clusters.append(basin_list)

    return clusters

def draw_with_metric(basin_list, metric, clusters):
    """绘制流域地图，根据 clusters 着色，并用散点大小表示 metric 大小
    Parameters:
     - basin_list: list[str], 流域编号列表
     - metric: list[float], 对应流域的指标值（顺序与 basin_list 对应）
     - clusters: list[list[str]], 包含 6 个子列表，每个子列表为一个聚类的流域编号
    """
    dataset_path = Path('../data/CAMELS_US/')           # 数据集路径
    basin_metadata = pd.read_csv(dataset_path / 'basin_metadata/gauge_information.txt', 
                                 index_col='GAGE_ID', sep='\t', engine='python', 
                                 dtype={'GAGE_ID': str, 'HUC_02': str})
    basin_metadata = basin_metadata.loc[basin_list]     # 筛选流域数据
    gdf = gpd.GeoDataFrame(basin_metadata, 
                           geometry=gpd.points_from_xy(basin_metadata["LONG"], basin_metadata["LAT"]), 
                           crs="EPSG:4326")     # 转换为 GeoDataFrame
    gdf = gdf.to_crs(epsg=3857)                 # 重新投影到 Web Mercator (EPSG:3857) 以适配底图

    x = gdf.geometry.x
    y = gdf.geometry.y

    metric_values = np.array(metric)            # 转换 metric 为 numpy 数组
    metric_values[metric_values < -1] = -1      # 将小于 -1 的值设为 -1 (针对 NSE)

    # 为每个流域分配聚类 ID
    cluster_ids = np.zeros(len(basin_list), dtype=int)  # 初始化聚类 ID 数组
    for i, basin in enumerate(basin_list):
        for cluster_idx, cluster in enumerate(clusters):
            if basin in cluster:
                cluster_ids[i] = cluster_idx
                break

    # 定义颜色（为 6 个聚类分配不同颜色）
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    if len(clusters) > len(colors):
        raise ValueError("聚类数量超过预定义颜色数量，请扩展 colors 列表")

    # 根据 metric 值计算散点大小（归一化到 50-500 范围）
    sizes = 50 + (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min() + 1e-6) * 450

    # 创建绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制散点，按聚类着色，大小反映 metric
    for cluster_idx in range(len(clusters)):
        mask = cluster_ids == cluster_idx
        ax.scatter(x[mask], y[mask], s=sizes[mask], c=colors[cluster_idx], 
                   label=f'类别 {cluster_idx + 1}', edgecolor="black", alpha=0.8)

    # 添加数值标注
    for i in range(len(basin_list)):
        ax.text(x.iloc[i], y.iloc[i], f"{metric_values[i]:.2f}", fontsize=12, 
                ha="right", va="bottom", color="black")

    # 添加底图 (高德矢量地图)
    tile_url = "http://wprd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=7&x={x}&y={y}&z={z}"
    ctx.add_basemap(ax, source=tile_url, alpha=0.5)

    # 自定义图例：固定大小的圆圈
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'类别 {i + 1}', 
                              markerfacecolor=colors[i], markersize=10, markeredgecolor='black') 
                       for i in range(len(clusters))]
    ax.legend(handles=legend_elements, loc='upper right')

    # 移除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()

if __name__ == '__main__':
    clusters = load_clusters()
    basin_list = load_basin_list('../data/basin_list/30_basin_list_evenly_test.txt')

    generalization = 'space'
    regional_lstm_transformer_log = './logs/' + generalization + '/regional-lstm-transformer.log'
    lstm_log = './logs/' + generalization + '/lstm.log'
    gru_log = './logs/' + generalization + '/gru.log'
    lstm_transformer_log = './logs/' + generalization + '/lstm-transformer.log'
    handoff_lstm_log = './logs/' + generalization + '/handoff-lstm.log'

    metric = load_log(regional_lstm_transformer_log)[0]   # 只取 NSE
    metric = np.delete(metric, -1)
    
    draw_with_metric(basin_list, metric, clusters)  # 绘制流域分布图