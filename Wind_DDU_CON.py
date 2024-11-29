from netCDF4 import Dataset
import earthaccess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import warnings
import json
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Login and download data
auth = earthaccess.login()
results = earthaccess.search_data(
    doi='10.5067/VJAFPLI1CSIV',
    temporal=("2019-11-25", "2020-01-25"),
)
files = earthaccess.download(results, "./MERRA2_Downloads")

all_U2M = []
all_V2M = []

for file in files:
    data = Dataset(file, mode='r')
    
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    
    lon_mask = (lons >= 120) & (lons <= 145)
    lat_mask = (lats >= -76) & (lats <= -66)
    
    lons_subset = lons[lon_mask]
    lats_subset = lats[lat_mask]
    
    U2M = data.variables['U2M'][:, lat_mask, lon_mask]
    V2M = data.variables['V2M'][:, lat_mask, lon_mask]
    
    FillValueU2M = U2M.fill_value
    FillValueV2M = V2M.fill_value
    U2M = np.ma.masked_equal(U2M, FillValueU2M).filled(np.nan)
    V2M = np.ma.masked_equal(V2M, FillValueV2M).filled(np.nan)
    
    all_U2M.append(U2M)
    all_V2M.append(V2M)

U2M = np.concatenate(all_U2M, axis=0)
V2M = np.concatenate(all_V2M, axis=0)

lon, lat = np.meshgrid(lons_subset, lats_subset)

ws = np.sqrt(U2M**2 + V2M**2)

ws_min = np.nanmin(ws)
ws_max = np.nanmax(ws)
clevs = np.linspace(ws_min, ws_max, 30)  # 30 levels

with open(r'C:PLACEHOLDERPLACEHOLDERPLACEHOLDER', 'r') as f:
    waypoints = json.load(f)

# Setup base 
base_datetime = datetime(2019, 11, 25, 0, 0)

fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

cf_initial = ax.contourf(lon, lat, ws[0,:,:], levels=clevs, transform=ccrs.PlateCarree(), cmap=plt.cm.viridis)
plt.colorbar(cf_initial, ax=ax, label='Wind Speed (m/s)')

def update(frame):
    ax.clear()
    ax.set_extent([120, 145, -76, -66])
    ax.coastlines(resolution="50m", linewidth=1)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='black', linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator([120, 125, 130, 135, 140, 145])
    gl.ylocator = mticker.FixedLocator([-76, -72, -68, -66])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':12, 'color':'black'}
    gl.ylabel_style = {'size':12, 'color':'black'}
    
    current_datetime = base_datetime + timedelta(hours=frame)
    

    cf = ax.contourf(lon, lat, ws[frame,:,:], levels=clevs, transform=ccrs.PlateCarree(), cmap=plt.cm.viridis)
    qv = ax.quiver(lon, lat, U2M[frame,:,:], V2M[frame,:,:], scale=420, color='k')
    
    # waypoints
    for feature in waypoints['features']:
        if feature['geometry']['type'] == 'MultiLineString':
            for line in feature['geometry']['coordinates']:
                lons, lats = zip(*line)
                ax.plot(lons, lats, 'r-', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    
    # timestamping
    ax.set_title(f'MERRA-2 2m Wind Speed and Direction\n{current_datetime.strftime("%Y-%m-%d %H:%M")}', size=14)
    
    return cf, qv

num_frames = U2M.shape[0]  # This should now be the total number of hours
anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/24)  # 24 fps, change it maybe

writer = animation.FFMpegWriter(fps=24, bitrate=10000)

# Save the animation
print(f"Saving...")
anim.save('test_HQ.mp4', writer=writer)
print("we done")