# 绘制不同模型在不同类型（共6类）的流域上表现的柱状图
import re
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rc('font', family='sans-serif', size=12)  # 设置字体和字号

def load_basin_list(file_path: Path):
    """读取流域列表"""
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

def mean_nse_group_by_cluster(basin_list, metrics, clusters):
    """
    Calculate the mean NSE for each cluster.
    """
    cluster_metrics = {}
    for basin, metric in zip(basin_list, metrics):
        for i, cluster in enumerate(clusters):
            if basin in cluster:
                if i not in cluster_metrics:
                    cluster_metrics[i] = []
                cluster_metrics[i].append(metric)  # 只取 NSE
    # 将负值替换为 0
    for i in cluster_metrics:
        cluster_metrics[i] = [max(0, nse) for nse in cluster_metrics[i]]
    return {i: np.mean(metrics) for i, metrics in cluster_metrics.items()}

if __name__ == '__main__':
    clusters = load_clusters()
    basin_list = load_basin_list('../data/basin_list/30_basin_list_evenly_test.txt')

    generalization = 'space'
    regional_lstm_transformer_log = './logs/' + generalization + '/regional-lstm-transformer.log'
    lstm_log = './logs/' + generalization + '/lstm.log'
    gru_log = './logs/' + generalization + '/gru.log'
    lstm_transformer_log = './logs/' + generalization + '/lstm-transformer.log'
    handoff_lstm_log = './logs/' + generalization + '/handoff-lstm.log'

    model_names = ['regional-lstm-transformer', 'lstm', 'gru', 'lstm-transformer', 'handoff-lstm']
    log_list = [regional_lstm_transformer_log, lstm_log, gru_log, lstm_transformer_log, handoff_lstm_log]

    metrics = []
    for log in log_list:
        metric = load_log(log)
        metrics.append(metric[0])   # 只取 NSE
    
    # 为每个模型计算每个聚类的平均 NSE
    cluster_metrics = {}
    for i, metric in enumerate(metrics):
        cluster_metrics[model_names[i]] = mean_nse_group_by_cluster(basin_list, metric, clusters)
    print(cluster_metrics)

    n_models = len(model_names)  # 5 models
    cluster_ids = list(range(6))  # Clusters 0 to 5

    # Set bar width and x-axis positions
    bar_width = 0.15
    x_positions = np.arange(len(cluster_ids))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each model
    for i, model in enumerate(model_names):
        # Calculate offset to position bars side by side
        offset = (i - n_models // 2) * bar_width
        # Extract NSE values for this model across all clusters
        values = [cluster_metrics[model][cluster] for cluster in cluster_ids]
        ax.bar(x_positions + offset, values, bar_width, label=model)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'类别 {i + 1}' for i in cluster_ids])
    ax.set_xlabel('流域类别')
    ax.set_ylabel('平均NSE')
    ax.set_title('不同模型在不同类型流域上的平均NSE')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    ax.grid(True, axis='y')  # Add horizontal grid lines

    # 调整布局以避免图例被裁剪
    plt.tight_layout()
    plt.show()
