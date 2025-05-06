# 流域列表说明

 - `516_basins_hourly.txt`: 代表以一小时为单位的数据集中包含的所有流域列表，即`usgs-streamflow-nldas_hourly.nc`中包含的516个流域
 - `531_basin_list.txt`: 普遍使用的531个流域
 - `huc_671/Region_xx.txt`: 代表HUC编号为xx的区域中所有的流域列表, 总数为671, 为原始的CAMELS US数据集包含的671个流域
 - `huc_516/Region_xx.txt`: 代表HUC编号为xx的区域中所有的basin列表, 总数为516, 为`usgs-streamflow-nldas_hourly.nc`中所含的流域
 - `clustered_basins/cluster_xx.txt`：代表KMeans聚类后，每个类中有哪些流域
 - `10_basin_list.txt`：从516个流域中随机抽取的10个流域
 - `30_basin_list_evenly.txt`：在六类流域（KMeans聚类）中，每个类别中随机抽取5个流域，组成的30个流域
 - `30_basin_list_evenly_test.txt`：在六类流域（KMeans聚类）中，每个类别中随机抽取5个流域，组成的30个流域，与`30_basin_list_evenly.txt`不重复
 - `150_basin_list_evenly.txt`：六类流域（KMeans聚类）中，每个类别中随机抽取25个流域，组成的150个流域