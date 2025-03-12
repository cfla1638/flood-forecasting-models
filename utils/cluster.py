# 对所有的CAMELS US流域进行聚类分析，并将不同类别的流域分别保存到不同的文件中
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from typing import List
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

os.environ["OMP_NUM_THREADS"] = "3" # 解决User Warning
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> <level>{message}</level>")

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

def load_camels_us_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """读取流域的静态属性
    Parameters:
     - data_dir: 文件夹 "CAMELS_US" 的路径
     - basins: 要加载的流域编号列表, 如果未传入该参数, 默认返回所有的流域

    Return:
     - pd.DataFrame: index为gauge_id
    """
    logger.info(f"Loading static attributes...")
    attributes_path = data_dir / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        logger.error(f"Attribute folder not found at {attributes_path}")
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')
        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    # convert huc column to double digit strings, 例如，1会被转换为'01'
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    # 根据传入的流域编号列表筛选数据
    if basins:
        if any(b not in df.index for b in basins):
            logger.error('Some basins are missing static attributes.')
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]
        logger.info(f"Static attributes loaded for {len(basins)} basins.")
    else:
        logger.info('Static attributes loaded for all basins.')

    logger.info(f"Static attributes loaded.")

    return df

# 聚类分析主函数
def cluster_camels_basins(data_dir: Path, basins: List[str] = [], max_clusters: int = 20):
    """
    对CAMELS US流域进行聚类，自动选择最佳集群数目。
    
    Parameters:
    - data_dir: CAMELS数据集目录
    - basins: 要分析的流域列表，默认空则分析所有流域
    - max_clusters: 最大尝试的集群数目
    
    Returns:
    - labels: 每个流域的集群标签
    - optimal_k: 最佳集群数目
    """
    # 1. 加载数据
    df = load_camels_us_attributes(data_dir, basins)
    
    # 2. 选择数值属性
    numeric_attrs = ['p_mean',
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
    df_numeric = df[numeric_attrs].fillna(df[numeric_attrs].mean())
    
    logger.info(f"Selected {len(numeric_attrs)} numeric attributes for clustering.")
    
    # 3. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)
    
    # 4. PCA降维（保留90%方差）
    pca = PCA(n_components=0.9)  # 保留90%方差
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"Reduced to {X_pca.shape[1]} dimensions with PCA, explaining {sum(pca.explained_variance_ratio_):.2%} variance.")
    
    # 5. 自动选择最佳集群数目
    silhouette_scores = []
    wss = []  # Within-cluster sum of squares
    k_range = range(2, min(max_clusters + 1, len(X_pca) // 15 + 1))  # 确保每个集群至少15个样本
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_pca)

        # 计算轮廓分数
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append(score)
        
        # 计算WSS
        wss.append(kmeans.inertia_)
        
        logger.info(f"k={k}, Silhouette Score={score:.3f}, WSS={kmeans.inertia_:.2f}")
    
    # 选择最佳k（轮廓分数最高）
    optimal_k = k_range[np.argmax(silhouette_scores)]
    logger.info(f"Optimal number of clusters: {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")
    
    # 6. 使用最佳k进行最终聚类
    kmeans_final = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    labels = kmeans_final.fit_predict(X_pca)
    
    # 将标签添加到原始数据
    df_numeric['cluster'] = labels
    
    # 7. 可视化（轮廓分数和肘部图）
    plt.figure(figsize=(12, 5))
    
    # 轮廓系数
    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.axvline(x=optimal_k, color='r', linestyle='--')
    plt.title('类簇数目与轮廓系数的关系')
    plt.xlabel('类簇数目(k)')
    plt.ylabel('轮廓系数(Silhouette Score)')
    
    # 肘部图
    plt.subplot(1, 2, 2)
    plt.plot(k_range, wss, marker='o')
    plt.axvline(x=optimal_k, color='r', linestyle='--')
    plt.title('肘部图')
    plt.xlabel('类簇数目 (k)')
    plt.ylabel('误差平方和(Within-Cluster Sum of Squares, SSE)')
    
    plt.tight_layout()
    plt.show()
    
    # # 8. 输出每个集群的属性均值
    # cluster_summary = df_numeric.groupby('cluster')[numeric_attrs].mean()
    # logger.info("Cluster summary (mean values):")
    # print(cluster_summary)
    
    return df_numeric['cluster'], optimal_k

def save_clusters_to_files(data_dir: Path, output_dir: Path, basins: List[str] = [], max_clusters: int = 10):
    """
    根据聚类结果将不同类别的流域分别保存到不同的文件中，每行一个gauge id。
    
    Parameters:
    - data_dir: CAMELS 数据集目录
    - output_dir: 输出文件目录
    - basins: 要分析的流域列表
    - max_clusters: 最大集群数
    """
    # 1. 执行聚类分析
    labels, optimal_k = cluster_camels_basins(data_dir, basins, max_clusters)
    
    # 2. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 将gauge id 按类别写入文件
    for cluster in range(optimal_k):
        cluster_file = output_dir / f"cluster_{cluster}.txt"
        
        # 筛选当前类别的流域
        cluster_ids = [basins[i] for i in range(len(basins)) if labels.iloc[i] == cluster]
        
        # 写入文件
        with open(cluster_file, 'w') as f:
            for gauge_id in cluster_ids:
                f.write(f"{gauge_id}\n")
        
        print(f"Saved {len(cluster_ids)} basins to {cluster_file}")



# 示例用法
if __name__ == "__main__":
    data_dir = Path("../data/CAMELS_US")
    basins = load_basin_list(Path('../data/basin_list/516_basins_hourly.txt'))
    # output_dir = Path("../data/basin_list/clustered_basins")
    # save_clusters_to_files(data_dir, output_dir, basins)
    cluster_camels_basins(data_dir, basins)