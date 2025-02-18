import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import contextily as ctx

dataset_path = Path('../data/CAMELS_US/')      # 数据集路径

basin_metadata = pd.read_csv(dataset_path / 'basin_metadata/gauge_information.txt', index_col='GAGE_ID', sep='\t', engine='python', dtype={'GAGE_ID': str, 'HUC_02': str})

# 筛选 HUC_02
# basin_metadata = basin_metadata[basin_metadata['HUC_02'] == '03']

# 转换为 GeoDataFrame
gdf = gpd.GeoDataFrame(basin_metadata, geometry=gpd.points_from_xy(basin_metadata["LONG"], basin_metadata["LAT"]), crs="EPSG:4326")

# 创建绘图
fig, ax = plt.subplots(figsize=(8, 6))

# 重新投影到 Web Mercator (EPSG:3857) 以适配底图
gdf = gdf.to_crs(epsg=3857)

# 绘制站点
gdf.plot(ax=ax, color="red", markersize=50, alpha=0.8, edgecolor="black", label="Gage Stations")

# 添加底图
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)

# 设置图例和标题
ax.legend()
ax.set_title("Gage Station Locations", fontsize=12)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# 移除坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])

# 保存为高分辨率图片（适用于论文）
# plt.savefig("gage_map.svg", format="svg", dpi=300)  # 矢量格式
# plt.savefig("gage_map.pdf", format="pdf", dpi=300)  # 论文优选格式
# plt.savefig("gage_map.png", format="png", dpi=600)  # 高分辨率位图

# 显示地图
plt.show()