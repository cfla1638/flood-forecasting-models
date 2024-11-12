import xarray as xr

dataset = xr.open_dataset('../data/CAMELS_US/hourly/usgs-streamflow-nldas_hourly.nc')
print(dataset)