import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import contextily as ctx

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

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

def draw_basins(basin_list):
    """绘制流域地图
    Parameter:
     - basin_list: 流域列表
    """
    dataset_path = Path('../data/CAMELS_US/')      # 数据集路径
    basin_metadata = pd.read_csv(dataset_path / 'basin_metadata/gauge_information.txt', index_col='GAGE_ID', sep='\t', engine='python', dtype={'GAGE_ID': str, 'HUC_02': str})

    # 筛选位于流域列表中的站点
    basin_metadata = basin_metadata.loc[basin_list]

    # 转换为 GeoDataFrame
    gdf = gpd.GeoDataFrame(basin_metadata, geometry=gpd.points_from_xy(basin_metadata["LONG"], basin_metadata["LAT"]), crs="EPSG:4326")

    # 创建绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 重新投影到 Web Mercator (EPSG:3857) 以适配底图
    gdf = gdf.to_crs(epsg=3857)

    # 绘制站点
    gdf.plot(ax=ax, color="red", markersize=50, alpha=0.8, edgecolor="black", label="Gauge Stations")

    tile_url = "http://wprd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=7&x={x}&y={y}&z={z}"
    ctx.add_basemap(ax, source=tile_url, alpha=0.5)

    # 设置图例和标题
    ax.legend()
    ax.set_title("Gauge Station Locations", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # 移除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 显示地图
    plt.show()

def draw_with_metric(basin_list, metric):
    """绘制流域地图，并根据 metric 以热力图颜色显示
    Parameters:
     - basin_list: list[str], 流域编号列表
     - metric: list[float], 对应流域的指标值（顺序与 basin_list 对应）
    """
    dataset_path = Path('../data/CAMELS_US/')           # 数据集路径
    basin_metadata = pd.read_csv(dataset_path / 'basin_metadata/gauge_information.txt', index_col='GAGE_ID', sep='\t', engine='python', dtype={'GAGE_ID': str, 'HUC_02': str})
    basin_metadata = basin_metadata.loc[basin_list]     # 筛选流域数据
    gdf = gpd.GeoDataFrame(basin_metadata, 
                           geometry=gpd.points_from_xy(basin_metadata["LONG"], basin_metadata["LAT"]), 
                           crs="EPSG:4326")     # 转换为 GeoDataFrame
    gdf = gdf.to_crs(epsg=3857)                 # 重新投影到 Web Mercator (EPSG:3857) 以适配底图

    x = gdf.geometry.x
    y = gdf.geometry.y

    metric_values = np.array(metric)            # 转换 metric 为 numpy 数组
    metric_values[metric_values < -1] = -1      # 将小于 -1 的值设为 -1 (针对NSE)

    # 颜色映射
    cmap = plt.cm.coolwarm  # 选用颜色映射
    norm = plt.Normalize(vmin=metric_values.min(), vmax=metric_values.max())
    colors = cmap(norm(metric_values))

    # 创建绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制散点
    sc = ax.scatter(x, y, c=metric_values, cmap=cmap, norm=norm, s=80, edgecolor="black", alpha=0.8)

    # 添加数值标注
    for i in range(len(basin_list)):
        ax.text(x.iloc[i], y.iloc[i], f"{metric_values[i]:.2f}", fontsize=8, ha="right", va="bottom", color="black")

    # 添加底图(高德矢量地图)
    tile_url = "http://wprd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=7&x={x}&y={y}&z={z}"
    ctx.add_basemap(ax, source=tile_url, alpha=0.5)

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical")
    cbar.set_label("纳什效率系数 (NSE)")

    ax.set_title("流域分布和其纳什效率系数", fontsize=12)   # 设置标题

    # 移除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()

def plot_predictions(y_true, y_pred, scatter_color='blue', fit_color=None):
    """
    绘制实际值（x 轴）与预测值（y 轴）的散点图，并添加：
    1. 红色 1:1 直线（表示完美预测）
    2. 预测值与实际值的拟合直线
    
    参数：
    - y_true: 实际值（numpy 数组或列表）
    - y_pred: 预测值（numpy 数组或列表）
    - scatter_color: 散点的颜色（默认 'blue'）
    - fit_color: 拟合直线的颜色（默认与散点相同）
    """
    if fit_color is None:
        fit_color = scatter_color  # 默认与散点颜色相同

    # 计算 1:1 直线的范围
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    # 计算拟合直线
    coef = np.polyfit(y_true, y_pred, 1)  # 一次线性拟合
    poly1d_fn = np.poly1d(coef)  # 转换为函数

    # 绘制散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, color=scatter_color, alpha=0.6, label="预测值")

    # 绘制 1:1 直线
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="1:1 线")

    # 绘制拟合直线
    plt.plot([min_val, max_val], poly1d_fn([min_val, max_val]), color=fit_color, linestyle='-', label="拟合线")

    # 图例和标签
    plt.xlabel("观测流量值")
    plt.ylabel("预测流量值")
    plt.legend()
    plt.title("观测值与预测值的散点图")

    plt.show()



if __name__ == '__main__':
    # basin_list_path = '../data/basin_list/train_val_test/Region_03_train.txt'  # 流域列表路径
    # basin_list = load_basin_list(Path(basin_list_path))
    # metrics = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1]
    # draw_with_metric(basin_list, metrics)
    
    # basin_list = '../data/basin_list/32_basin_list.txt'
    # basin_list = load_basin_list(Path(basin_list))
    # draw_basins(basin_list)

    # 生成一些示例数据
    # np.random.seed(42)
    # y_true = np.linspace(0, 100, 50)  # 实际值
    # y_pred = y_true * 0.9 + np.random.normal(0, 5, 50)  # 模拟预测值（带噪声）

    # # 绘制
    # plot_predictions(y_true, y_pred, scatter_color='blue')
    basin_list = '../data/basin_list/30_basin_list_evenly.txt'
    basin_list = load_basin_list(Path(basin_list))
    draw_basins(basin_list)