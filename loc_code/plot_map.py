"""
Created on 2025/11/24

@author: Minyu Guo

This code is designed to generate localization result maps by converting predicted epicentral distances and
back-azimuths into geographical coordinates (latitude/longitude) and plotting them on the map.
"""
from mpl_toolkits.basemap import Basemap
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from geopy.distance import geodesic
from geographiclib.geodesic import Geodesic
import pygmt

geod = Geodesic.WGS84

receive = np.load('./3JAPAN_test/stations_location.npy', allow_pickle=True)
# pre_dist = np.load('./k_net_test2/predict_dist.npy')
# # pre_dist_uncertainty = np.load('./k_net_test_uncertainty/predict_dist.npy')
# pre_azi = np.load('./k_net_test2/predict_azimuth_reg.npy')
pre_dist = np.load('./acc_vel_test3/predict_dist.npy')
# pre_dist_uncertainty = np.load('./k_net_test_uncertainty/predict_dist.npy')
pre_azi = np.load('./acc_vel_test3/predict_azimuth_reg.npy')
print(pre_azi.shape, pre_dist.shape)


# pre_dist = np.load('./low_snr_uncertainty/predict_dist.npy')
# # pre_dist_uncertainty = np.load('./k_net_test_uncertainty/predict_dist.npy')
# pre_p_travel = np.load('./low_snr_uncertainty/predict_p_travel_time.npy')
# pre_azi = np.load('./low_snr_uncertainty/predict_azimuth_reg.npy')
# print(pre_azi.shape, pre_dist.shape)
#
# label_dist = np.load('./comparing/low_loc_test_set_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/low_loc_test_set_p_travel.npy').squeeze().reshape(-1)
# label_azi = np.load('./comparing/low_loc_test_set_azimuth.npy').squeeze().reshape(-1, 2)


# fig, ax = plt.subplots(figsize=(10, 6))
# # ax.set_title(f'P-travel time residual (s): {res:.4f}', fontsize=10)
# position = ax.get_position()
# main_map = Basemap(projection='cyl', resolution='f', lat_0=so_la, lon_0=so_lo,
#                    llcrnrlon=so_lo - 1.5, llcrnrlat=so_la - 1.5, urcrnrlon=so_lo + 1.5, urcrnrlat=so_la + 1.5)
#
# main_map.drawcoastlines(linewidth=1, color='#D2B48C')
# main_map.drawcountries(linewidth=1, color='#D2B48C')
# main_map.drawrivers(linewidth=1, color='#D2B48C')
# # main_map.drawstates(linewidth=0.5, color='red')
# main_map.drawmapboundary(fill_color='#6495ED')
# main_map.fillcontinents(color='#E6E6FA', lake_color='#6495ED')
# main_map.drawparallels(np.arange(-90., 91., 1), labels=[True, False, False, True], linewidth=0.2)
# main_map.drawmeridians(np.arange(-180, 181, 1), labels=[False, False, False, True], linewidth=0.2)
# so_x, so_y = main_map(so_lo, so_la)
# for i in range(receive.shape[0]):
#     re_la = receive[i, 0]
#     re_lo = receive[i, 1]
#
#     # min_azi = min_azimuth[i]
#     # max_azi = max_azimuth[i]
#     # mean_azi = pre_azi[i]
#     pre_azimuth_deg = pre_azi[i]
#     # pre_angles_rad = math.atan2(pre_azi[i, 1], pre_azi[i, 0])
#     # pre_azimuth_deg = math.degrees(pre_angles_rad)
#     # if pre_azimuth_deg < 0:
#     #     pre_azimuth_deg+=360
#     # print(pre_azimuth_deg)
#     # print(mean_azi, min_azi, max_azi, sd_p_azi[i])
#     r = math.sqrt((re_lo - so_lo) ** 2 + (re_la - so_la) ** 2)
#     # if  max_d > 0.8:
#     #     continue
#     # if abs(max_d-r)/r > 0.2:
#     #     continue
#
#     result = geod.Direct(re_la, re_lo, pre_azimuth_deg, pre_dist[i] * 1000)  # 转换为米
#     epicenter_lat = result['lat2']
#     epicenter_lon = result['lon2']
#
#     result = geod.Inverse(re_la, re_lo, so_la, so_lo)
#     real_azimuth = result['azi1']
#     real_distance = result['s12']/1000
#     if real_azimuth < 0:
#         real_azimuth = real_azimuth + 360
#     error = abs(real_azimuth - pre_azimuth_deg)
#     # if error > 360:
#     #     pre_azimuth_deg -= 360
#
#
#     re_x, re_y = main_map(re_lo, re_la)
#     if i == 0:
#         main_map.scatter(re_x, re_y, marker='^', color='black', s=200, zorder=5, label='Receiver')
#         circle = patches.Circle((re_x, re_y), pre_dist[i]/100, linewidth=1, edgecolor='r', facecolor='none', zorder=5)
#         plt.gca().add_patch(circle)
#         plt.legend(loc='upper left', fontsize=15)
#     else:
#         main_map.scatter(re_x, re_y, marker='^', color='black', s=200, zorder=5)
#         circle = patches.Circle((re_x, re_y), pre_dist[i] / 100, linewidth=1, edgecolor='r', facecolor='none', zorder=5)
#         plt.gca().add_patch(circle)
#
#
# main_map.scatter(so_x, so_y, marker='*', color='yellow', s=200, zorder=5, label='Source')
# plt.legend(loc='upper left', fontsize=15)
#
# plt.savefig(f'./figure_plot/GRSL/review/3k_net_test.png', format='png', transparent=False,
#             dpi=500)
# plt.show()
    # inp = input("Press a key to plot the next waveform!")
    # if inp == "r":
    #     continue

azi_error = []
dist_error = []
hro_error = []
for i in range(receive.shape[0]):
    re_la = receive[i, 0]
    re_lo = receive[i, 1]
    so_la = 34.975
    so_lo = 138.213


    pre_angles_rad = math.atan2(pre_azi[i, 1], pre_azi[i, 0])
    pre_azimuth_deg = math.degrees(pre_angles_rad)
    if pre_azimuth_deg < 0:
        pre_azimuth_deg+=360
    # print(pre_azimuth_deg)
    # print(mean_azi, min_azi, max_azi, sd_p_azi[i])
    r = math.sqrt((re_lo - so_lo) ** 2 + (re_la - so_la) ** 2)
    # if  max_d > 0.8:
    #     continue
    # if abs(max_d-r)/r > 0.2:
    #     continue

    result = geod.Direct(re_la, re_lo, pre_azimuth_deg, pre_dist[i] * 1000)  # 转换为米
    epicenter_lat = result['lat2']
    epicenter_lon = result['lon2']

    result = geod.Inverse(re_la, re_lo, so_la, so_lo)
    real_azimuth = result['azi1']
    real_distance = result['s12']/1000

    if real_azimuth < 0:
        real_azimuth = real_azimuth + 360
    error = abs(real_azimuth - pre_azimuth_deg)
    # if error > 360:
    #     pre_azimuth_deg -= 360
    if error > 90:
        continue

    pre_result = geod.Inverse(re_la, re_lo, epicenter_lat, epicenter_lon)
    pre_hro_error = pre_result['s12']/1000
    hro_error.append(pre_hro_error)
    azi_error.append(error)
    dist_error.append(abs(real_distance - pre_dist[i]))

    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.set_title(f'P-travel time residual (s): {res:.4f}', fontsize=10)
    position = ax.get_position()
    main_map = Basemap(projection='cyl', resolution='c', lat_0=re_la, lon_0=re_lo,
                       llcrnrlon=re_lo - 1.5, llcrnrlat=re_la - 1.5, urcrnrlon=re_lo + 1.5, urcrnrlat=re_la + 1.5)

    main_map.drawcoastlines(linewidth=1, color='#D2B48C')
    main_map.drawcountries(linewidth=1, color='#D2B48C')
    main_map.drawrivers(linewidth=1, color='#D2B48C')
    # main_map.drawstates(linewidth=0.5, color='red')
    main_map.drawmapboundary(fill_color='#6495ED')
    main_map.fillcontinents(color='#E6E6FA', lake_color='#6495ED')
    import matplotlib.patches as mpatches

    main_map.drawparallels(np.arange(-90., 91., 1), labels=[True, False, False, True], linewidth=0.2)
    main_map.drawmeridians(np.arange(-180, 181, 1), labels=[False, False, False, True], linewidth=0.2)
    so_x, so_y = main_map(so_lo, so_la)
    # azi = pre_azi[i]
    # if dist > 50 and dist < 60:
    #     continue
    # if abs(distance - dist)/distance > 0.2 :
    #     print(abs(distance - dist)/distance)
    #     continue

    # pre_angles_rad = math.atan2(pre_azi[i, 1], pre_azi[i, 0])
    # pre_azimuth_deg = math.degrees(pre_angles_rad)
    # pre_t = pre_time[i]
    epicenter_x, epicenter_y = main_map(epicenter_lon, epicenter_lat)
    main_map.scatter(epicenter_x, epicenter_y, marker='*', color='r', s=200, zorder=5, label='Pre_epicenter')

    re_x, re_y = main_map(re_lo, re_la)
    main_map.scatter(re_x, re_y, marker='^', color='black', s=200, zorder=5, label='Receiver')
    circle = patches.Circle((re_x, re_y), pre_dist[i]/75, linewidth=1, edgecolor='r', facecolor='none', zorder=5)
    plt.gca().add_patch(circle)
    plt.legend(loc='upper left', fontsize=15)

    main_map.scatter(so_x, so_y, marker='*', color='yellow', s=200, zorder=5, label='Source')
    plt.legend(loc='upper left', fontsize=15)

    # wedge = mpatches.Wedge((re_x, re_y), max_d,
    #                        min_azi+90, max_azi+90,
    #                        facecolor='red', alpha=0.5,
    #                        edgecolor='darkred', linewidth=2)
    #
    # ax.add_patch(wedge)
    # plt.savefig(f'./figure_plot/GRSL/review/low_snr_test_{i}.png', format='png', transparent=False,
    #             dpi=500)
    # plt.show()
    # inp = input("Press a key to plot the next waveform!")
    # if inp == "r":
    #     continue

print(np.mean(dist_error), np.mean(azi_error), np.mean(hro_error))

