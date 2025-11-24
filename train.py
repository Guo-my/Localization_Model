"""
Created on 2025/11/24

@author: Minyu Guo

This code reads data from the STEAD dataset to train a localization model. The input data will be automatically
segmented and augmented. For detailed information, please refer to the other code files or contact guomy5339@163.com.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

The code utilizes CPU multiprocessing to read data and reduce processing time, which may consume significant computational resources.
Please run with caution on personal computers!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
import numpy as np
import torch
import os
import torch.nn as nn
import seisbench
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm
import argparse
import random
import matplotlib.pyplot as plt
from Read_data import read, load_data
from multiprocessing import Pool
from azimuth_nn import our_bazi_net
from classify_azimuth_nn import class_our_bazi_net
from our_dist_nn import our_dist
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges, TestSignals, issq_cwt
from ssqueezepy.visuals import plot, imshow
import math
random.seed(3407)


def bins2deg(bins, nb_bins, nb_ensb):
    sz_bins = 360 / nb_bins
    bin_shft = nb_bins / 2

    deg_ensb1 = np.mod((bins + [i / nb_ensb for i in range(nb_ensb)] + 0.5) * sz_bins, 360)
    deg1 = np.mean(deg_ensb1, 1)

    deg_ensb2 = (np.mod(bins + bin_shft, nb_bins) + [i / nb_ensb for i in range(nb_ensb)] + 0.5) * sz_bins
    deg2 = np.mod(np.mean(deg_ensb2, 1) - bin_shft * sz_bins, 360)

    return np.where(np.var(deg_ensb1, 1) < np.var(deg_ensb2, 1), deg1, deg2)



class CrossEntropyLossWithProbs(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithProbs, self).__init__()
    def forward(self, prob, target_probs):
        loss = -torch.sum(target_probs * torch.log(prob))
        return loss.mean()


file_name = "../STEAD/merged/merge.hdf5"
csv_file = "../STEAD/merged/merge.csv"
# training_data, validation, noise = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=True, mode='train')
# test, realdata_test = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=False, mode='test')
training_data, validation = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=True, mode='train')
test = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=False, mode='test')


def test_process_batch(i):
    one_batch_clean, one_batch_d, one_batch_p, one_batch_s, one_batch_dist, one_batch_p_travel, one_batch_deep, one_batch_azimuth = test.__getitem__(i)
    return (np.array(one_batch_clean), np.array(one_batch_d),
            np.array(one_batch_p), np.array(one_batch_s), np.array(one_batch_dist),
            np.array(one_batch_p_travel), np.array(one_batch_deep), np.array(one_batch_azimuth))


def val_process_batch(i):
    one_batch_clean, one_batch_d, one_batch_p, one_batch_s, one_batch_dist, one_batch_p_travel, one_batch_deep, one_batch_azimuth = validation.__getitem__(i)
    return (np.array(one_batch_clean), np.array(one_batch_d),
            np.array(one_batch_p), np.array(one_batch_s), np.array(one_batch_dist),
            np.array(one_batch_p_travel), np.array(one_batch_deep), np.array(one_batch_azimuth))


def train_process_batch(i):
    one_batch_clean, one_batch_d, one_batch_p, one_batch_s, one_batch_dist, one_batch_p_travel, one_batch_deep, one_batch_azimuth = training_data.__getitem__(i)
    return (np.array(one_batch_clean), np.array(one_batch_d),
            np.array(one_batch_p), np.array(one_batch_s), np.array(one_batch_dist),
            np.array(one_batch_p_travel), np.array(one_batch_deep), np.array(one_batch_azimuth))


# def noise_process_batch(i):
#     one_batch_noise, _, _, _, _, _, _ = noise.__getitem__(i)
#     return np.array(one_batch_noise)


def main(opt):
    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    OUT_PATH = opt.out_path
    LR = opt.lr
    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("GPU is not available.")
    print("epochs:", NUM_EPOCHS)
    print("batch_size:", BATCH_SIZE)
    print("out_path:", OUT_PATH)
    print("lr:", LR)
    print('Begin to process the validation set')
    print('-----------------------------------------------------------------------------------------------------------')
    pool = Pool()
    val_length = int(len(validation))
    val_results = list(
        tqdm(pool.imap(val_process_batch, range(val_length)), total=val_length))
    validation_c, validation_d, validation_p, validation_s, validation_dist, validation_p_travel, validation_deep, validation_azimuth = [list(t) for t in zip(*val_results)]
    print(validation_c[1].shape)

    print('-----------------------------------------------------------------------------------------------------------')
    # test_length = int(len(test))
    # test_results = list(
    #     tqdm(pool.imap(test_process_batch, range(test_length)), total=test_length))
    # test_c, test_d, test_p, test_s, test_dist, test_p_travel, test_deep, test_azimuth = [list(t) for t in zip(*test_results)]
    # print(test_c[1].shape)
    # np.save('./comparing/test_set_raw', test_c)
    # np.save('./comparing/test_set_d', test_d)
    # np.save('./comparing/test_set_p', test_p)
    # np.save('./comparing/test_set_s', test_s)
    # np.save('./comparing/test_set_dist', test_dist)
    # np.save('./comparing/test_set_p_travel', test_p_travel)
    # np.save('./comparing/test_set_deep', test_deep)
    # np.save('./comparing/test_set_azimuth', test_azimuth)


    print('Begin to process the training data')
    print('-----------------------------------------------------------------------------------------------------------')
    train_length = int(len(training_data))
    results = list(
        tqdm(pool.imap(train_process_batch, range(train_length)), total=train_length))
    c, d, p, s, dist, p_travel, deep, azimuth = [list(t) for t in zip(*results)]
    # comb_c = c + c
    # comb_d = d + d
    # comb_p = p + p
    # comb_s = s + s

    indices = list(range(len(c)))
    random.shuffle(indices)
    c_shuffled = [c[i] for i in indices]
    dist_shuffled = [dist[i] for i in indices]
    p_travel_shuffled = [p_travel[i] for i in indices]
    deep_shuffled = [deep[i] for i in indices]
    azimuth_shuffled = [azimuth[i] for i in indices]
    print('All the data are processed over!')

    start_time = time.time()

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set save path
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    argsDict = opt.__dict__
    with open(OUT_PATH + "/hyperparameter.txt", "w") as f:
        f.writelines("-" * 10 + "start" + "-" * 10 + "\n")
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " :    " + str(value) + "\n")
        f.writelines("-" * 10 + "end" + "-" * 10)

    model_path = os.path.join(OUT_PATH, "crnn")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    statistics_path = os.path.join(OUT_PATH, "statistics")
    if not os.path.exists(statistics_path):
        os.makedirs(statistics_path)

    loss_path = os.path.join(OUT_PATH, "train_loss")
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)

    # model = class_our_bazi_net()
    # model = our_bazi_net()
    model = our_dist()
    model.weights_init()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {total_params} trainable parameters")

    # initialize
    # optimizer = torch.optim.SGD(DAEmodel.parameters(), lr=LR, momentum=MOMENTUM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8)
    loss_mse = nn.MSELoss()

    save_metric = 0.0
    train_loss = []
    val_loss = []

    train_pick_loss = []
    train_p_travel_loss = []
    train_dist_loss = []
    train_deep_loss = []
    train_azi_loss = []

    val_pick_loss = []
    val_p_travel_loss = []
    val_dist_loss = []
    val_azi_loss = []
    val_deep_loss = []

    # train_length = int(len(training_data))

    # train
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(range(len(c_shuffled)), colour="white")
        running_results = {'batch_sizes': 0, "loss": 0, "pick_loss": 0, "p_travel_loss": 0, "dist_loss": 0, "deep_loss": 0, "azi_loss": 0}

        model.train()
        for i in train_bar:
            label_x = torch.tensor(c_shuffled[i], dtype=torch.float32).permute(0, 2, 1)
            label_dist = torch.tensor(dist_shuffled[i], dtype=torch.float32)
            label_p_travel = torch.tensor(p_travel_shuffled[i], dtype=torch.float32)
            # label_deep = torch.tensor(deep_shuffled[i], dtype=torch.float32)
            # label_azimuth = torch.tensor(azimuth_shuffled[i], dtype=torch.float32)


            label_x = label_x.to(device)
            label_dist = label_dist.to(device)
            label_p_travel = label_p_travel.to(device)
            # label_deep = label_deep.to(device)
            # label_azimuth = label_azimuth.to(device)
            batch_size = label_x.size(0)
            running_results["batch_sizes"] += batch_size
            output_dist, output_p_travel = model(label_x)

            # azimuth
            # output_azi = model(label_x)

            # Calculate the loss value of Dist-travel model
            loss_dist = loss_mse(output_dist, label_dist)
            loss_p_travel = loss_mse(output_p_travel, label_p_travel)

            # Calculate the loss value of classification model(Back-azi model)
            # loss1 = loss_cel(output_azi[:, 0, :], label_azimuth[:, 0])
            # loss2 = loss_cel(output_azi[:, 1, :], label_azimuth[:, 1])
            # loss3 = loss_cel(output_azi[:, 2, :], label_azimuth[:, 2])
            # loss4 = loss_cel(output_azi[:, 3, :], label_azimuth[:, 3])
            # loss5 = loss_cel(output_azi[:, 4, :], label_azimuth[:, 4])
            # loss6 = loss_cel(output_azi[:, 5, :], label_azimuth[:, 5])
            # loss7 = loss_cel(output_azi[:, 6, :], label_azimuth[:, 6])
            # loss8 = loss_cel(output_azi[:, 7, :], label_azimuth[:, 7])
            # loss9 = loss_cel(output_azi[:, 8, :], label_azimuth[:, 8])
            # loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9)/9

            loss = loss_dist + loss_p_travel
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_results["loss"] += loss.item() * batch_size
            # running_results["dist_loss"] += loss_dist.item() * batch_size
            # running_results["p_travel_loss"] += loss_p_travel.item() * batch_size
            # running_results["deep_loss"] += loss_deep.item() * batch_size
            # running_results["azi_loss"] += loss_azi.item() * batch_size

            train_bar.set_description(desc='[%d/%d] loss: %.7f' % (
                epoch, NUM_EPOCHS, running_results["loss"] / running_results["batch_sizes"]
            ))
        train_loss.append(running_results["loss"] / running_results["batch_sizes"])
        train_pick_loss.append(running_results["pick_loss"] / running_results["batch_sizes"])
        train_p_travel_loss.append(running_results["p_travel_loss"] / running_results["batch_sizes"])
        train_dist_loss.append(running_results["dist_loss"] / running_results["batch_sizes"])
        train_deep_loss.append(running_results["deep_loss"] / running_results["batch_sizes"])
        train_azi_loss.append(running_results["azi_loss"] / running_results["batch_sizes"])

        model.eval()
        with torch.no_grad():
            val_bar = tqdm(range(len(validation)), colour="white")
            valing_results = {'batch_sizes': 0, "loss": 0, "pick_loss": 0, "p_travel_loss": 0, "dist_loss": 0, "deep_loss": 0, "azi_loss": 0}
            dist_error = []
            p_travel_error = []
            deep_error = []
            azi_error = []
            for i in val_bar:
                val_c = torch.tensor(validation_c[i], dtype=torch.float32).permute(0, 2, 1)
                val_dist = torch.tensor(validation_dist[i], dtype=torch.float32)
                val_p_travel = torch.tensor(validation_p_travel[i], dtype=torch.float32)
                # val_azimuth = torch.tensor(validation_azimuth[i], dtype=torch.float32)

                val_c = val_c.to(device)
                val_dist = val_dist.to(device)
                val_p_travel = val_p_travel.to(device)
                # val_deep = val_deep.to(device)
                # val_azimuth = val_azimuth.to(device)
                batch_size = val_c.size(0)
                valing_results["batch_sizes"] += batch_size
                output_dist, output_p_travel = model(val_c)
                # azimuth
                # output_azi = model(val_c)

                # out_azi = np.array(torch.argmax(nn.functional.softmax(output_azi.cpu(), dim=2), dim=2))
                # print(out_azi[0])
                # print(val_azimuth[0])
                # deg = bins2deg(out_azi, 4, 9)
                # deg_label = bins2deg(np.array(val_azimuth.cpu()), 4, 9)
                # dist_error.append((np.sum(np.abs(deg_label - deg)) / 128).item())

                # You can use the following code to monitor the results of validation set!!!
                dist_error.append((torch.sum(torch.abs(val_dist - output_dist))/128).item())
                # p_travel_error.append((torch.sum(torch.abs(val_p_travel - output_p_travel))/128).item())
                # deep_error.append((torch.sum(torch.abs(val_deep - output_deep)) / 128).item())
                # azi_error.append((torch.sum(torch.abs((torch.atan2(val_azimuth[:, 1], val_azimuth[:, 0]) -
                #                                       torch.atan2(output_azi[:, 1], output_azi[:, 0])) * (180 / torch.pi))) / 128).item())

                loss_dist = loss_mse(output_dist, val_dist)
                loss_p_travel = loss_mse(output_p_travel, val_p_travel)

                # loss1 = loss_cel(output_azi[:, 0, :], val_azimuth[:, 0])
                # loss2 = loss_cel(output_azi[:, 1, :], val_azimuth[:, 1])
                # loss3 = loss_cel(output_azi[:, 2, :], val_azimuth[:, 2])
                # loss4 = loss_cel(output_azi[:, 3, :], val_azimuth[:, 3])
                # loss5 = loss_cel(output_azi[:, 4, :], val_azimuth[:, 4])
                # loss6 = loss_cel(output_azi[:, 5, :], val_azimuth[:, 5])
                # loss7 = loss_cel(output_azi[:, 6, :], val_azimuth[:, 6])
                # loss8 = loss_cel(output_azi[:, 7, :], val_azimuth[:, 7])
                # loss9 = loss_cel(output_azi[:, 8, :], val_azimuth[:, 8])
                # loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9) / 9

                loss = loss_dist + loss_p_travel

                valing_results["loss"] += loss.item() * batch_size
                # valing_results["pick_loss"] += (loss_d * 0.05 + loss_p * 0.45 + loss_s * 0.5).item() * batch_size
                # valing_results["dist_loss"] += loss_dist.item() * batch_size
                # valing_results["p_travel_loss"] += loss_p_travel.item() * batch_size
                # valing_results["deep_loss"] += loss_deep.item() * batch_size
                # valing_results["azi_loss"] += loss_azi.item() * batch_size

                val_bar.set_description(desc='[val] loss: %.7f' % (
                        valing_results["loss"] / valing_results["batch_sizes"]))

        val_loss.append(valing_results["loss"] / valing_results["batch_sizes"])
        val_pick_loss.append(valing_results["pick_loss"] / valing_results["batch_sizes"])
        val_p_travel_loss.append(valing_results["p_travel_loss"] / valing_results["batch_sizes"])
        val_dist_loss.append(valing_results["dist_loss"] / valing_results["batch_sizes"])
        val_deep_loss.append(valing_results["deep_loss"] / valing_results["batch_sizes"])
        val_azi_loss.append(valing_results["azi_loss"] / valing_results["batch_sizes"])

        if epoch > 0 and epoch % 10 == 0:
            fig1, ax1 = plt.subplots()
            ax1.plot(train_loss, color='#bb3f3f', label='Train_loss', marker='s', markersize=4)
            ax1.legend()
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            plt.show()

            fig1, ax1 = plt.subplots()
            ax1.plot(val_loss, color='#665fd1', label='Val_loss', marker='o', markersize=4)
            ax1.legend()
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            plt.show()

            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_pick_loss, color='#bb3f3f', label='Train_pick_loss', marker='s', markersize=4)
            # ax1.plot(val_pick_loss, color='#665fd1', label='Val_pick_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()
            #
            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_dist_loss, color='#bb3f3f', label='Train_dist_loss', marker='s', markersize=4)
            # ax1.plot(val_dist_loss, color='#665fd1', label='Val_dist_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()
            #
            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_p_travel_loss, color='#bb3f3f', label='Train_p_travel_loss', marker='s', markersize=4)
            # ax1.plot(val_p_travel_loss, color='#665fd1', label='Val_p_travel_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()
            # #
            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_deep_loss, color='#bb3f3f', label='Train_deep_loss', marker='s', markersize=4)
            # ax1.plot(val_deep_loss, color='#665fd1', label='Val_deep_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()
            #
            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_azi_loss, color='#bb3f3f', label='Train_azi_loss', marker='s', markersize=4)
            # ax1.plot(val_azi_loss, color='#665fd1', label='Val_azi_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()


        # update the LR
        scheduler.step()

        # save model parameters
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), model_path + "/detect_test_netpar_epoch_%d.pth" % epoch)

        metric = 1/(valing_results["loss"]/valing_results["batch_sizes"])
        if metric > save_metric:
            torch.save(model.state_dict(), model_path + "/detect_test_best.pth")
            save_metric = metric
            print("save new model")

        print('mean dist error', sum(dist_error)/len(validation_c))
        print('mean p travel time error', sum(p_travel_error) / len(validation_c))
        print('mean deep error', sum(deep_error) / len(validation_c))
        print('mean azi error', sum(azi_error) / len(validation_c))

    np.save(os.path.join(loss_path, 'loss_train'), train_loss)
    np.save(os.path.join(loss_path, 'loss_pick_train'), train_pick_loss)
    np.save(os.path.join(loss_path, 'loss_dist_train'), train_dist_loss)
    np.save(os.path.join(loss_path, 'loss_p_travel_train'), train_p_travel_loss)
    np.save(os.path.join(loss_path, 'loss_deep_train'), train_deep_loss)
    np.save(os.path.join(loss_path, 'loss_azi_train'), train_azi_loss)
    np.save(os.path.join(loss_path, 'loss_val'), val_loss)
    np.save(os.path.join(loss_path, 'loss_pick_val'), val_pick_loss)
    np.save(os.path.join(loss_path, 'loss_dist_val'), val_dist_loss)
    np.save(os.path.join(loss_path, 'loss_p_travel_val'), val_p_travel_loss)
    np.save(os.path.join(loss_path, 'loss_deep_val'), val_deep_loss)
    np.save(os.path.join(loss_path, 'loss_azi_val'), val_azi_loss)

    fig1, ax1 = plt.subplots()
    ax1.plot(train_loss, color='#bb3f3f', label='Train_loss', marker='s', markersize=4)
    ax1.plot(val_loss, color='#665fd1', label='Val_loss', marker='o', markersize=4)
    ax1.legend()
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.show()

    # fig1, ax1 = plt.subplots()
    # ax1.plot(train_pick_loss, color='#bb3f3f', label='Train_pick_loss', marker='s', markersize=4)
    # ax1.plot(val_pick_loss, color='#665fd1', label='Val_pick_loss', marker='o', markersize=4)
    # ax1.legend()
    # ax1.set_title('Training Loss')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss')
    # plt.show()

    # fig1, ax1 = plt.subplots()
    # ax1.plot(train_dist_loss, color='#bb3f3f', label='Train_dist_loss', marker='s', markersize=4)
    # ax1.plot(val_dist_loss, color='#665fd1', label='Val_dist_loss', marker='o', markersize=4)
    # ax1.legend()
    # ax1.set_title('Training Loss')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss')
    # plt.show()
    #
    # fig1, ax1 = plt.subplots()
    # ax1.plot(train_p_travel_loss, color='#bb3f3f', label='Train_p_travel_loss', marker='s', markersize=4)
    # ax1.plot(val_p_travel_loss, color='#665fd1', label='Val_p_travel_loss', marker='o', markersize=4)
    # ax1.legend()
    # ax1.set_title('Training Loss')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss')
    # plt.show()
    #
    # fig1, ax1 = plt.subplots()
    # ax1.plot(train_deep_loss, color='#bb3f3f', label='Train_deep_loss', marker='s', markersize=4)
    # ax1.plot(val_deep_loss, color='#665fd1', label='Val_deep_loss', marker='o', markersize=4)
    # ax1.legend()
    # ax1.set_title('Training Loss')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss')
    # plt.show()
    #
    # fig1, ax1 = plt.subplots()
    # ax1.plot(train_azi_loss, color='#bb3f3f', label='Train_azi_loss', marker='s', markersize=4)
    # ax1.plot(val_azi_loss, color='#665fd1', label='Val_azi_loss', marker='o', markersize=4)
    # ax1.legend()
    # ax1.set_title('Training Loss')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss')
    # plt.show()
    print(f'min loss in train: {min(train_loss):.7f}')
    print(f'min loss in validation: {min(val_loss):.7f}')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print('running time: %dmin' % elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Localization Model")
    parser.add_argument("--num_epochs", default=50, type=int, help="train epoch number")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate of model")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size of train dataset")
    parser.add_argument("--out_path", default="./location_model_param", type=str, help="the path of save file")
    opt = parser.parse_args()
    main(opt)
