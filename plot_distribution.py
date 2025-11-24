import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert
from sklearn.metrics import r2_score

def caculate_prob(data):
    prob = np.zeros((9, 360))
    for i in range(9):
        prob[i, i * 10:i * 10 + 90] = data[i, 0]
        prob[i, i * 10 + 90:i * 10 + 180] = data[i, 1]
        prob[i, i * 10 + 180:i * 10 + 270] = data[i, 2]
        prob[i, i * 10 + 270:360] = data[i, 3]
        prob[i, 0:i * 10] = data[i, 3]
    p = np.mean(prob, axis=0)
    max_prob = np.max(p)
    return max_prob

def moving_average(data, window_size):
    output = []
    sd = []
    for i in range(data.shape[0]//window_size):
        output.append(np.mean(data[i*window_size:(i+1)*window_size]))
        sd.append(np.std(data[i*window_size:(i+1)*window_size]))
    return output, sd
# sigma_p_plot = np.load('./our_model_ps_distribution_highsnr/sigma_p_plot.npy')
# sigma_s_plot = np.load('./our_model_ps_distribution_highsnr/sigma_s_plot.npy')
# sigma_p_plot = np.abs(sigma_p_plot)
# sigma_s_plot = np.abs(sigma_s_plot)
# min_edge = min(np.min(sigma_p_plot), np.min(sigma_s_plot))
# max_edge = max(np.max(sigma_p_plot), np.max(sigma_s_plot))
# num = int((max_edge - min_edge) / 0.01) + 1
#
# bin_edges = np.linspace(min_edge, max_edge, num)
# bin_counts1, _ = np.histogram(sigma_p_plot, bins=bin_edges)
# bin_counts2, _ = np.histogram(sigma_s_plot, bins=bin_edges)
#
# mae_p = 0.0132
# sigma_p = 0.0193
# mae_s = 0.0291
# sigma_s = 0.0393
#
# plt.bar(bin_edges[:-1], bin_counts1, width=np.diff(bin_edges), edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.4f \n SD=%.4f' % (mae_p, sigma_p))
# plt.legend(fontsize=10)
# # plt.xlim(-0.2, 0.2)
# plt.xlabel('Time residuals (s)')
# plt.ylabel('Number of Picks')
# plt.title('P-phase picks')
# # plt.savefig(f'./figure_plot/pick_plot/Our_p_time_residual_high.png', format='png', dpi=500)
# plt.show()
#
# plt.bar(bin_edges[:-1], bin_counts2, width=np.diff(bin_edges), edgecolor="white", align="edge",
#         color="#FF6347", label='MAE=%.4f \n SD=%.4f' % (mae_s, sigma_s))
# plt.legend(fontsize=10)
# # plt.xlim(-0.2, 0.2)
# plt.xlabel('Time residuals (s)')
# plt.ylabel('Number of Picks')
# plt.title('S-phase picks')
# # plt.savefig(f'./figure_plot/pick_plot/Our_s_time_residual_high.png', format='png', dpi=500)
# plt.show()


label_azi = np.load('./comparing/new_high_loc_test_set_azimuth.npy').squeeze().reshape(-1, 2)

# label_dist = np.load('./comparing/low_loc_uncertainty_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/low_loc_uncertainty_p_travel.npy').squeeze().reshape(-1)
# label_azi = np.load('./comparing/low_loc_uncertainty_azimuth.npy').squeeze().reshape(-1, 2)
# label_uncertainty = np.load('./comparing/low_loc_uncertainty.npy').squeeze().reshape(-1)
# print(min(label_uncertainty))
# label_snr = np.load('./comparing/low_loc_uncertainty_snr.npy').squeeze().reshape(-1, 3)

# pre_dist_result = np.load('./low_loc_uncertainty_result/predict_dist.npy')
# pre_p_travel_result = np.load('./low_loc_uncertainty_result/predict_p_travel_time.npy')
# pre_azimuth_class = np.load('./low_loc_uncertainty_result/predict_azimuth.npy')
pre_azimuth_class_sof = np.load('./new_high_loc/predict_azimuth_class.npy')
pre_azimuth_reg = np.load('./new_high_loc/predict_azimuth_reg.npy')
print(pre_azimuth_reg.shape)

# mean_snr = np.mean(label_snr, axis=1)
# sorted_indices = np.argsort(mean_snr)

label_angles_rad = np.arctan2(label_azi[:, 1], label_azi[:, 0])
label_azi_deg = np.degrees(label_angles_rad)
label_azi_deg[label_azi_deg < 0] += 360

pre_azimuth_reg = np.arctan2(pre_azimuth_reg[:, 1], pre_azimuth_reg[:, 0])
pre_azimuth_reg = np.degrees(pre_azimuth_reg)
pre_azimuth_reg[pre_azimuth_reg < 0] += 360

# plot_azi_deg_class = pre_azimuth_class - label_azi_deg
plot_azi_deg_reg = pre_azimuth_class_sof - label_azi_deg

# plot_azi_deg_reg[plot_azi_deg_reg > 180] -= 360
# plot_azi_deg_reg[plot_azi_deg_reg < -180] += 360

error_index = [index for index, error in enumerate (np.abs(plot_azi_deg_reg)) if error > 170]
azi = label_azi_deg[error_index]

plt.figure(figsize=(10, 6))
plt.hist(azi, bins=50, alpha=0.6, label='Horizontal uncertainty', edgecolor="white")
plt.legend(fontsize=10)
plt.xlabel('Horizontal error (km)', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/event_uncertainty.png', format='png', transparent=True, dpi=500)
plt.show()

plot_azi_deg_class[plot_azi_deg_class > 180] -= 360
plot_azi_deg_class[plot_azi_deg_class < -180] += 360
sorted_indices_class_error = np.argsort(abs(plot_azi_deg_class))
print(max(abs(plot_azi_deg_class)), min(abs(plot_azi_deg_class)))
pre_azimuth_class_sof = pre_azimuth_class_sof[sorted_indices_class_error]
prob = [caculate_prob(class_soft) for class_soft in pre_azimuth_class_sof]
smoothed_prob, smoothed_prob_sd = moving_average(np.array(prob), window_size=2000)
# plt.figure(figsize=(10, 6))
# x = np.arange(0, 180, 180 / len(smoothed_prob))
# plt.plot(x, smoothed_prob, label='Classification error', marker='s', markersize=4, color='#4169E1')
# plt.errorbar(x, smoothed_prob, yerr=smoothed_prob_sd, linestyle='', capsize=5, ecolor='#4169E1')
# plt.legend()
# plt.xlabel("Error (deg)", fontsize=15)
# plt.ylabel("Probability (%)", fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/prob&error.png', format='png', transparent=True, dpi=500)
# plt.show()

plot_azi_deg_reg[plot_azi_deg_reg > 180] -= 360
plot_azi_deg_reg[plot_azi_deg_reg < -180] += 360

class_azimuth_error = plot_azi_deg_class[sorted_indices]
reg_azimuth_error = plot_azi_deg_reg[sorted_indices]
abs_class_azimuth_error = abs(class_azimuth_error)
abs_reg_azimuth_error = abs(reg_azimuth_error)
class_azimuth_percent = abs_class_azimuth_error/180
reg_azimuth_percent = abs_reg_azimuth_error/180

# mean_error_azi = np.mean(plot_azi_deg_class)
# mae_azi = np.mean(np.abs(plot_azi_deg_class))
# sigma_azi = np.std(plot_azi_deg_class)

mean_error_azi = np.mean(plot_azi_deg_reg)
mae_azi = np.mean(np.abs(plot_azi_deg_reg))
sigma_azi = np.std(plot_azi_deg_reg)

n_bins = 30
bin_width = 360 / n_bins  # 固定宽度
bin_edges = np.linspace(-180, 180, n_bins + 1)

# bin_counts1, bin_edges1 = np.histogram(plot_azi_deg_reg, bins=bin_edges)
# plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.2f \n SD=%.2f' % (mae_azi, sigma_azi))
# plt.legend(fontsize=15)
# plt.xlim(-180, 180)
# plt.xlabel('Backazimuth Residuals (deg)', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/review/reg_our_azi_model_residual_low.png', format='png', transparent=True, dpi=500)
# plt.show()
#
dist_r_squared = r2_score(label_azi_deg, pre_azimuth_class)
plt.figure(figsize=(10, 6))
plt.scatter(label_azi_deg, pre_azimuth_class, color='#4169E1', alpha=0.7, s=5)
plt.plot([0, 360], [0, 360], color='white', linestyle='--', linewidth=3)
# plt.plot([0, 180], [180, 360], color='red', linestyle='--', linewidth=3)
# plt.plot([180, 360], [0, 180], color='red', linestyle='--', linewidth=3)
plt.text(100, 300, f'R² = {dist_r_squared:.2f}', fontsize=15, bbox=dict(facecolor='white'))
plt.xlabel('True Backazimuth (deg)', fontsize=15)
plt.ylabel('Predicted Backazimuth (deg)', fontsize=15)
# plt.legend()
plt.grid()
# plt.xlim(0, 150)
# plt.ylim(0, 150)
# plt.savefig(f'./figure_plot/GRSL/review/R2class_our_azi_model_scatter_low.png', format='png', transparent=True, dpi=500)
plt.show()


# # ---------------------------------------------------------------------------------------------------------
plot_dist = pre_dist_result - label_dist
plot_p_travel = pre_p_travel_result - label_p_travel

mean_error_dist = np.mean(plot_dist)
mae_dist = np.mean(np.abs(plot_dist))
sigma_dist = np.std(plot_dist)

mean_error_p_travel = np.mean(plot_p_travel)
mae_p_travel = np.mean(np.abs(plot_p_travel))
sigma_p_travel = np.std(plot_p_travel)

n_bins = 30
bin_width = 80 / n_bins  # 固定宽度
bin_edges = np.linspace(-40, 40, n_bins + 1)

# bin_counts1, bin_edges1 = np.histogram(plot_dist, bins=bin_edges)
# plt.bar(bin_edges1[:-1], bin_counts1, width=bin_width, edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.2f \n SD=%.2f' % (mae_dist, sigma_dist))
# plt.legend(fontsize=15)
# plt.xlim(-40, 40)
# plt.xlabel('Distance Residuals (km)', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/review/our_dist_residual_low.png', format='png', transparent=True, dpi=500)
# plt.show()
#
# n_bins = 30
# bin_width = 10 / n_bins  # 固定宽度
# bin_edges = np.linspace(-5, 5, n_bins + 1)
#
# bin_counts1, bin_edges1 = np.histogram(plot_p_travel, bins=bin_edges)
# plt.bar(bin_edges1[:-1], bin_counts1, width=bin_width, edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.2f \n SD=%.2f' % (mae_p_travel, sigma_p_travel))
# plt.legend(fontsize=15)
# plt.xlim(-5, 5)
# plt.xlabel('Time Residuals (s)', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/review/our_time_residual_low.png', format='png', transparent=True, dpi=500)
# plt.show()
# # # # ---------------------------------------------------------------------------------------------------------
# dist_r_squared = r2_score(label_dist, pre_dist_result)
# plt.figure(figsize=(10, 6))
# plt.scatter(label_dist, pre_dist_result, color='#4169E1', alpha=0.7, s=5)
# plt.plot([0, 110], [0, 110], color='white', linestyle='--', linewidth=3)
# plt.text(0, 100, f'R² = {dist_r_squared:.2f}', fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
# plt.xlabel('True Distance (km)', fontsize=15)
# plt.ylabel('Predicted Distance (km)', fontsize=15)
# # plt.legend()
# plt.grid()
# # plt.xlim(0, 150)
# # plt.ylim(0, 150)
# plt.savefig(f'./figure_plot/GRSL/review/our_dist_scatter_low.png', format='png', transparent=True, dpi=500)
# plt.show()
# #
# #
# p_travel_r_squared = r2_score(label_p_travel, pre_p_travel_result)
# plt.figure(figsize=(10, 6))
# plt.scatter(label_p_travel, pre_p_travel_result, color='#4169E1', alpha=0.7, s=5)
# plt.plot([0, 30], [0, 30], color='white', linestyle='--', linewidth=3)
# plt.text(0, 20, f'R² = {p_travel_r_squared:.2f}', fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
# plt.xlabel('True P-travel Time (s)', fontsize=15)
# plt.ylabel('Predicted P-travel Time (s)', fontsize=15)
# # plt.legend()
# plt.grid()
# current_xlim = plt.xlim()
# plt.xlim(current_xlim[0], 25)
# current_ylim = plt.ylim()
# plt.ylim(current_ylim[0], 25)
# plt.savefig(f'./figure_plot/GRSL/review/our_time_scatter_low.png', format='png', transparent=True, dpi=500)
# plt.show()
#----------------------------------------------------------------------------------------------------------------------
# plt.figure(figsize=(10, 6))
# plt.hist(label_uncertainty, bins=10, alpha=0.6, label='Horizontal uncertainty', edgecolor="white")
# plt.legend(fontsize=10)
# plt.xlabel('Horizontal error (km)', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/event_uncertainty.png', format='png', transparent=True, dpi=500)
# plt.show()

plot_dist = pre_dist_result - label_dist
plot_p_travel = pre_p_travel_result - label_p_travel

# plt.figure(figsize=(10, 6))
# plt.hist(label_snr, bins=10, alpha=0.6, label=['E component', 'N component', 'Z component'])
# plt.legend(fontsize=10)
# plt.xlabel('SNR (dB)', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/event_snr.png', format='png', transparent=True, dpi=500)
# plt.show()
def moving_average(data, window_size):
    output = []
    for i in range(data.shape[0]//window_size):
        output.append(np.mean(data[i*window_size:(i+1)*window_size]))
    return output, output

dist_error = plot_dist[sorted_indices]
time_error = plot_p_travel[sorted_indices]
dist = label_dist[sorted_indices]
time_p_travel = label_p_travel[sorted_indices]
abs_dist_error = abs(dist_error)
abs_time_error = abs(time_error)
dist_percent = abs_dist_error/(dist+1e-9)
time_percent = abs_time_error/(time_p_travel+1e-9)
time_percent = [1 if percent > 10 else percent for percent in time_percent]
smoothed_dist, smoothed_dist_sd = moving_average(abs_dist_error, window_size=2000)
smoothed_time, smoothed_time_sd = moving_average(abs_time_error, window_size=2000)
smoothed_dist_percent, smoothed_dist_percent_sd = moving_average(dist_percent, window_size=2000)
smoothed_time_percent, smoothed_time_percent_sd = moving_average(np.array(time_percent), window_size=2000)
smoothed_class_azi_percent, smoothed_class_azi_percent_sd = moving_average(np.array(class_azimuth_percent), window_size=2000)
smoothed_reg_azi_percent, smoothed_reg_azi_percent_sd = moving_average(np.array(reg_azimuth_percent), window_size=2000)
smoothed_class_azi, smoothed_class_azi_sd = moving_average(np.array(abs_class_azimuth_error), window_size=2000)
smoothed_reg_azi, smoothed_reg_azi_sd = moving_average(np.array(abs_reg_azimuth_error), window_size=2000)
#
x = np.arange(min(mean_snr), max(mean_snr), (max(mean_snr) - min(mean_snr)) / len(smoothed_dist))
plt.figure(figsize=(10, 6))

ax1 = plt.gca()
ax1.plot(x, smoothed_dist, label='Distance', marker='s', markersize=4, color='#4169E1')
# ax1.errorbar(x, smoothed_dist, yerr=smoothed_dist_sd, linestyle='', capsize=5, ecolor='#4169E1')
ax1.set_xlabel("SNR (dB)", fontsize=15)
ax1.set_ylabel("Distance Error (km)", fontsize=15, color='#4169E1')  # 设置左侧 Y 轴标签
ax1.tick_params(axis='y', labelcolor='#4169E1')  # 左侧 Y 轴刻度颜色

ax2 = ax1.twinx()
ax2.plot(x, smoothed_time, label='P_travel_time', marker='*', markersize=4, color='#bb3f3f')
# ax2.errorbar(x, smoothed_time, yerr=smoothed_time_sd, linestyle='', capsize=5, ecolor='#bb3f3f')
ax2.set_ylabel("Travel Time Error (s)", fontsize=15, color='#bb3f3f')  # 设置右侧 Y 轴标签
ax2.tick_params(axis='y', labelcolor='#bb3f3f')  # 右侧 Y 轴刻度颜色

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.savefig(f'./figure_plot/GRSL/1dist_error_line.png', format='png', transparent=True, dpi=500)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, smoothed_class_azi, label='Back_azimuth(Classification)', marker='v', markersize=4, color='#4169E1')
# plt.errorbar(x, smoothed_class_azi, yerr=smoothed_class_azi_sd, linestyle='', capsize=5, ecolor='#4169E1')
plt.plot(x, smoothed_reg_azi, label='Back_azimuth(Regression)', marker='X', markersize=4, color='#bb3f3f')
# plt.errorbar(x, smoothed_reg_azi, yerr=smoothed_reg_azi_sd, linestyle='', capsize=5, ecolor='#bb3f3f')
plt.legend()
plt.xlabel("SNR (dB)", fontsize=15)
plt.ylabel("Error (deg)", fontsize=15)
plt.savefig(f'./figure_plot/GRSL/1azimuth_error_line.png', format='png', transparent=True, dpi=500)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, smoothed_dist_percent, label='Distance', marker='s', markersize=4, color='#4169E1')
plt.plot(x, smoothed_time_percent, label='P_travel_time', marker='*', markersize=4, color='#bb3f3f')
plt.plot(x, smoothed_class_azi_percent, label='Back_azimuth(Classification)', marker='v', markersize=4, color='#f1f33f')
plt.plot(x, smoothed_reg_azi_percent, label='Back_azimuth(Regression)', marker='X', markersize=4, color='#0cdc73')
plt.legend()
plt.xlabel("SNR (dB)", fontsize=15)
plt.ylabel("Error (%)", fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/percent_error_line.png', format='png', transparent=True, dpi=500)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------