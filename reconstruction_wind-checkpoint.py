import matplotlib
import xarray as xr
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from joblib import Parallel, delayed
import sys

class mainfunc(object):
    def __init__(self, factors=None, mapsize=(2,2), year1=None, year2=None, season=None, month=None, months=None, interval=None, mkdata_flag=True):
        self.map_extent = [-125+360,-100+360,30,50]

        if mkdata_flag:
            xdata = xr.open_dataset(f'som/wind_ano_{season}_{mapsize[0]}x{mapsize[1]}.nc')['ano']
            print(xdata)

            distance = xr.open_dataset(f"reconstruct/distance/distance_{season}_{mapsize[0]}x{mapsize[1]}_facz.{interval}.nc")["distance"]
            weight   = 1. / distance * 1000000.

            if month is not None:
                xdata = xdata.sel(time=xdata["time.month"]==month)
            if months is not None:
                weight = weight.sel(time=weight["time.month"].isin(months))

            xweight  = weight.to_pandas()
            value = np.sort(xweight.values)[:,-4]
            for i in np.arange(xweight.shape[0]):
                xweight.iloc[i, xweight.iloc[i,:]<value[i]] = np.nan
#            print(xweight)
            weight = xr.Dataset.from_dataframe(xweight).to_array("bmu")
            weight = weight / weight.sum("bmu")
            print(weight)
#            print(weight.sum("bmu").to_pandas())

            if int(interval[:-1]) > 10:
                ydata = xdata * weight
                ydata = ydata.sum("bmu")
            else:
                ntime = weight.time.size
                i = 0
                ydata = []
                while True:
                    s = i * 500
                    e = s + 500
                    if e >= ntime: e = ntime
                    _ydata = xdata * weight.isel(time=slice(s, e))
                    ydata.append(_ydata.sum("bmu"))
                    i += 1
                    print(i, s, e, ntime)
                    if e == ntime: break
                ydata = xr.concat(ydata, dim='time')
            ydata.name = "ano_est"
            print(ydata)
           
#            mf = self.calc_wind
#            wind = Parallel(n_jobs=8)(delayed(mf)(year=year, 
#                xdata=xdata, weight=weight, mapsize=mapsize, season=season, interval=interval) 
#                for year in np.arange(year1, year2+1)
#            )
#            wind = xr.concat(wind, dim="time")
#            print(wind)
#
            self.save_load_data(factors, mapsize, data=[ydata], season=season, interval=interval, mode="save")
        else:
            diff, ndif, pval, mean = self.save_load_data(factors, mapsize, month=month, interval=interval, mode="load")

    def calc_wind(self, year=None, xmean=None, weight=None, mapsize=None, season=None, interval=None):
        print(f"year--{year}")
        print(xmean)
        xweight = weight.sel(time=weight["time.year"]==year)
        print(xweight)
        xwind = xmean * xweight
        xwind = xwind.sum("bmu").groupby(xwind["time.dayofyear"]) + clm
        print(clm)
        xwind.name = "wind_est"
        print(xwind)
        print("=================================================")
        xwind.to_netcdf(f"./reconstruct/estimated_wind_distance_{season}.{interval}.{year}.nc")
        return xwind

    def save_load_data(self, factors, mapsize, data=None, season=None, interval=None, mode=None):
        fname = f"./reconstruct/ano_est/estimated_wind_{season}_{mapsize[0]}x{mapsize[1]}.{interval}.nc"
        if mode == "save": 
#            data[0].name = "ano"
            data[0].name = "ano_est"
            xr.merge(data).to_netcdf(fname)
        if mode == "load":
            ds = xr.open_dataset(fname)
            return ds["ano"], ds["wind_est"]

if __name__ == "__main__":
    interval = int(sys.argv[1])
    season = sys.argv[2]
    if season=="cold": months=[10,11,12,1,2,3], 
    if season=="warm": months=[4,5,6,7,8,9], 

    mainfunc(factors=("z"), mapsize=(4,4), 
             year1=1981, year2=2020, 
             season=season, months=months,
             interval=f"{interval:02d}D",
             mkdata_flag=True)
