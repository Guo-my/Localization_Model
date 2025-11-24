"""
Created on 2025/11/24

@author: Minyu Guo

This code loads the trained model and its pre-trained weights to perform corresponding predictions.
Upon completion of the model prediction, error distribution histograms and scatter plots will be automatically generated.

"""
import numpy as np
import torch
import os
import torch.nn as nn
from data_augmentation import DataGenerator
import time
from our_dist_nn import our_dist
from classify_azimuth_nn import class_our_bazi_net
from azimuth_nn import our_bazi_net
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from Error import calculate_error
from retnet_train import bins2deg


def noise(noisedata):
    length = len(noisedata)
    num = 6000 // length
    noisesignal = np.empty(6000)
    for i in range(num):
        noisesignal[i*length:i*length + length] = noisedata[:]
    noisesignal[num*length:] = noisedata[:6000-num*length]
    return noisesignal


def calculate_snr(signal, noise):
    signal_power = np.sum(np.square(signal))+ 1e-8
    noise_power = np.sum(np.square(noise))+ 1e-8
    snr = 10 * np.log10(signal_power/ noise_power)
    return snr


def normalize(data, mode='std'):
    'Normalize waveforms in each batch'

    data -= np.mean(data, axis=2, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=2, keepdims=True)
        assert (max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data

    elif mode == 'std':
        std_data = np.std(data, axis=2, keepdims=True)
        std_data[std_data == 0] = 1
        data /= std_data
    return data


# This function is primarily used to partition the data into batches with a size of 128 and store them in a list.
def batch(data):
    batch_size = 128
    batch_data = []
    batch_num = data.shape[0]//batch_size + 1
    for i in range(batch_num):
        batch_data.append(data[i*batch_size:(i+1)*batch_size])
    return batch_data


data = np.load('./comparing/low_loc_uncertainty_raw.npy')
label_dist = np.load('./comparing/low_loc_uncertainty_dist.npy').squeeze().reshape(-1)
label_p_travel = np.load('./comparing/low_loc_uncertainty_p_travel.npy').squeeze().reshape(-1)
label_azi = np.load('./comparing/low_loc_uncertainty_azimuth.npy').squeeze().reshape(-1, 2)
label_uncertainty = np.load('./comparing/low_loc_uncertainty.npy').squeeze().reshape(-1)
label_snr = np.load('./comparing/low_loc_uncertainty_snr.npy').squeeze().reshape(-1, 3)

# K-Net data prediction
# E = np.load('./3JAPAN_test/acc_vel_test3/prompt_E_wave.npy')[:, :6000]
# N = np.load('./3JAPAN_test/acc_vel_test3/prompt_N_wave.npy')[:, :6000]
# Z = np.load('./3JAPAN_test/acc_vel_test3/prompt_Z_wave.npy')[:, :6000]
# print(E.shape)
# real_data_E = np.zeros_like(E)
# real_data_N = np.zeros_like(E)
# real_data_Z = np.zeros_like(E)
# real_data_E[:, 50:] = E[:, :5950]
# real_data_N[:, 50:] = N[:, :5950]
# real_data_Z[:, 50:] = Z[:, :5950]
# print(real_data_E.shape)
# real_data = np.stack((E, N, Z), axis=1)
# # real_data = np.stack((E[:, :6000], N[:, :6000], Z[:, :6000]), axis=1)
# # # real_data = np.load('./zhengyan/broadband.npy')
# print(real_data.shape)
# real_data = normalize(real_data)
# print(real_data.shape)
# batch_real_data = batch(real_data)
# print(len(batch_real_data))

# Loading the model to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = class_our_bazi_net()
# model = our_dist()
# model = our_bazi_net()
model = model.to(device)

# Loading the model parameters
model_file_path = './location_model_param/class_our_backazi/detect_test_best.pth'
if os.path.isfile(model_file_path):
    model.load_state_dict(torch.load(model_file_path))
    print("load parameters successfully")
else:
    print("no parameters")

# model.train()
model.eval()
pre_dist = []
pre_p_travel = []
pre_deep = []
pre_azi = []
batches = data.shape[0]
test_bar = tqdm(range(batches), colour="white")
with torch.no_grad():
    for i in test_bar:
        test_x = torch.tensor(data[i], dtype=torch.float32).permute(0, 2, 1)
        # test_x = torch.tensor(batch_real_data[i], dtype=torch.float32)

        test_x = test_x.to(device)
        # output_dist, output_p_travel = model(test_x)
        output_azi = model(test_x)

        # pre_dist.append(output_dist.cpu().numpy())
        # pre_p_travel.append(output_p_travel.cpu().numpy())
        out_azi = np.array(torch.argmax(nn.functional.softmax(output_azi.cpu(), dim=2), dim=2))
        deg = bins2deg(out_azi, 4, 9)
        pre_azi.append(deg)
        # pre_azi.append(output_azi.cpu().numpy())
        # one_reg_azi.append(output_azi.detach().cpu().numpy())
        # pre_azi.append(nn.functional.softmax(output_azi.cpu(), dim=2).numpy())
        # one_dist.append(output_dist.detach().cpu().numpy())
        # one_p.append(output_p_travel.detach().cpu().numpy())

# pre_dist_result = np.concatenate(pre_dist, axis=0).squeeze()
# pre_p_travel_result = np.concatenate(pre_p_travel, axis=0).squeeze()
pre_azimuth = np.concatenate(pre_azi, axis=0).squeeze()

save_path = './new_high_loc'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# np.save(os.path.join(save_path, 'predict_dist.npy'), pre_dist_result)
# np.save(os.path.join(save_path, 'predict_p_travel_time.npy'), pre_p_travel_result)
np.save(os.path.join(save_path, 'predict_azimuth_class.npy'), pre_azimuth)
print('Save')

# Epicentral distance error distribution histogram
# # # ---------------------------------------------------------------------------------------------------------
# plot_dist = pre_dist_result - label_dist
# # plot_p_travel = pre_p_travel_result - label_p_travel
#
# mean_error_dist = np.mean(plot_dist)
# mae_dist = np.mean(np.abs(plot_dist))
# sigma_dist = np.std(plot_dist)
#
# # mean_error_p_travel = np.mean(plot_p_travel)
# # mae_p_travel = np.mean(np.abs(plot_p_travel))
# # sigma_p_travel = np.std(plot_p_travel)
#
# n_bins = 30
# bin_width = 80 / n_bins
# bin_edges = np.linspace(-40, 40, n_bins + 1)
#
# bin_counts1, bin_edges1 = np.histogram(plot_dist, bins=bin_edges)
# plt.bar(bin_edges1[:-1], bin_counts1, width=bin_width, edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.2f \n SD=%.2f' % (mae_dist, sigma_dist))
# plt.legend(fontsize=15)
# plt.xlim(-40, 40)
# plt.xlabel('Distance Residuals (km)', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# # plt.savefig(f'./figure_plot/GRSL/review/mousavi_dist_residual_low.png', format='png', transparent=True, dpi=500)
# plt.show()
#
# n_bins = 30
# bin_width = 10 / n_bins
# bin_edges = np.linspace(-5, 5, n_bins + 1)
#
# bin_counts1, bin_edges1 = np.histogram(plot_p_travel, bins=bin_edges)
# plt.bar(bin_edges1[:-1], bin_counts1, width=bin_width, edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.2f \n SD=%.2f' % (mae_p_travel, sigma_p_travel))
# plt.legend(fontsize=15)
# plt.xlim(-5, 5)
# plt.xlabel('Time Residuals (s)', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# # plt.savefig(f'./figure_plot/GRSL/review/mousavi_time_residual_low.png', format='png', transparent=True, dpi=500)
# plt.show()

# Epicentral distance error distribution scatter plot
# # # ---------------------------------------------------------------------------------------------------------
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
# # plt.savefig(f'./figure_plot/GRSL/review/mousavi_dist_scatter_low.png', format='png', transparent=True, dpi=500)
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
# # plt.savefig(f'./figure_plot/GRSL/review/mousavi_time_scatter_low.png', format='png', transparent=True, dpi=500)
# plt.show()

# Back-azimuth error distribution histogram and scatter plot
# # # ---------------------------------------------------------------------------------------------------------
pre_angles_rad = np.arctan2(pre_azimuth[:, 1], pre_azimuth[:, 0])
pre_azimuth_deg = np.degrees(pre_angles_rad)
pre_azimuth_deg[pre_azimuth_deg < 0] += 360

label_angles_rad = np.arctan2(label_azi[:, 1], label_azi[:, 0])
label_azi_deg = np.degrees(label_angles_rad)
label_azi_deg[label_azi_deg < 0] += 360

plot_azi_deg = pre_azimuth_deg - label_azi_deg

# Mapping Errors to the 180-Degree Range
plot_azi_deg[plot_azi_deg > 180] -= 360
plot_azi_deg[plot_azi_deg < -180] += 360

mean_error_azi = np.mean(plot_azi_deg)
mae_azi = np.mean(np.abs(plot_azi_deg))
sigma_azi = np.std(plot_azi_deg)

n_bins = 30
bin_width = 360 / n_bins
bin_edges = np.linspace(-180, 180, n_bins + 1)

bin_counts1, bin_edges1 = np.histogram(plot_azi_deg, bins=bin_edges)
plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
        color="#4169E1", label='MAE=%.2f \n SD=%.2f' % (mae_azi, sigma_azi))
plt.legend(fontsize=15)
plt.xlim(-180, 180)
plt.xlabel('Backazimuth Residuals (deg)', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
# plt.savefig(f'./figure_plot/GRSL/review/class_our_azi_model_residual_high.png', format='png', transparent=True, dpi=500)
plt.show()

# dist_r_squared = r2_score(label_azi_deg, pre_azimuth_deg)
plt.figure(figsize=(10, 6))
plt.scatter(label_azi_deg, pre_azimuth_deg, color='#4169E1', alpha=0.7, s=5)
plt.plot([0, 360], [0, 360], color='white', linestyle='--', linewidth=3)
# plt.plot([0, 180], [180, 360], color='red', linestyle='--', linewidth=3)
# plt.plot([180, 360], [0, 180], color='red', linestyle='--', linewidth=3)
# plt.text(100, 300, f'R² = {dist_r_squared:.2f}', fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel('True Backazimuth (deg)', fontsize=15)
plt.ylabel('Predicted Backazimuth (deg)', fontsize=15)
# plt.legend()
plt.grid()
# plt.xlim(0, 150)
# plt.ylim(0, 150)
# plt.savefig(f'./figure_plot/GRSL/review/class_our_azi_model_scatter_high.png', format='png', transparent=True, dpi=500)
plt.show()

# The prediction back-azimuth from the classification model are visualized using a heatmap.
# # #--------------------------------------------------------------------------------------------------------------------
predict_data = [pre_azimuth[6689], pre_azimuth[8500], pre_azimuth[34470], pre_azimuth[3226], pre_azimuth[31272], pre_azimuth[37115]]
def caculate_prob(data):
    prob = np.zeros((9, 360))
    for i in range(9):
        prob[i, i * 10:i * 10 + 90] = data[i, 0]
        prob[i, i * 10 + 90:i * 10 + 180] = data[i, 1]
        prob[i, i * 10 + 180:i * 10 + 270] = data[i, 2]
        prob[i, i * 10 + 270:360] = data[i, 3]
        prob[i, 0:i * 10] = data[i, 3]
    p = np.mean(prob, axis=0)
    return p

i = 0
for azi in predict_data:
    i+=1
    p = caculate_prob(azi)
    theta = np.linspace(0, 2 * np.pi, 360)  # 0° ~ 360° → 0 ~ 2π
    r = np.array([0.9, 1.0])  # 半径范围（避免中心空白）
    T, R = np.meshgrid(theta, r)  # 生成网格
    Z = np.tile(p, (len(r), 1))  # 复制概率数据到二维

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    heatmap = ax.pcolormesh(T, R, Z, cmap='viridis', shading='auto')
    cbar = plt.colorbar(heatmap, ax=ax, label='Probability', pad=0.1, shrink=0.7)
    cbar.set_label('Probability', size=15)
    ax.set_yticks([])  # 隐藏半径刻度
    ax.set_title('Heatmap of Probability Distribution', pad=20, fontsize=15)
    ax.spines['polar'].set_visible(False)
    ax.set_theta_offset(np.pi / 2)  # 将 0° 旋转到北方（上方）
    ax.set_theta_direction(-1)
    ax.grid(visible=False)
    ax.tick_params(axis='x', labelsize=15)
    plt.savefig(f'./figure_plot/GRSL/review/low_snr_uncertainty_probability_{i}.png', format='png', transparent=True, dpi=500)
    plt.show()

