"""
Created on 2025/11/24

@author: Minyu Guo

This code is adapted from the original implementation provided by Mousavi et al. (2020).
This code provides a detailed explanation of how to segment the data and perform augmentation,
with each data augmentation method implemented through a dedicated function.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from obspy import Trace, Stream

# Convert angles to one-Hot labels
def deg2bins(deg, nb_bins, nb_ensb):
    sz_bins = 360 / nb_bins
    sz_shft = sz_bins / nb_ensb

    deg = np.expand_dims(deg, 1)
    bins = (np.repeat(deg, nb_ensb, 1) - [i * sz_shft for i in range(nb_ensb)]) // sz_bins
    bins = np.mod(bins, nb_bins).astype('int')

    return bins



class DataGenerator:
    def __init__(self,
                 list_IDs,
                 file_name,
                 dim,
                 batch_size=32,
                 n_channels=3,
                 phase_window=40,
                 shuffle=True,
                 norm_mode='std',
                 label_type='triangle',
                 augmentation=False,
                 add_event_r=None,
                 add_gap_r=None,
                 coda_ratio=1.4,
                 shift_event_r=None,
                 add_noise_r=None,
                 drop_channe_r=None,
                 scale_amplitude_r=None,
                 pre_emphasis=False):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.phase_window = phase_window
        self.list_IDs = list_IDs
        self.file_name = file_name
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.label_type = label_type
        self.augmentation = augmentation
        self.add_event_r = add_event_r
        self.add_gap_r = add_gap_r
        self.coda_ratio = coda_ratio
        self.shift_event_r = shift_event_r
        self.add_noise_r = add_noise_r
        self.drop_channe_r = drop_channe_r
        self.scale_amplitude_r = scale_amplitude_r
        self.pre_emphasis = pre_emphasis

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.augmentation:
            return 2 * int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.augmentation:
            indexes = self.indexes[index * self.batch_size // 2:(index + 1) * self.batch_size // 2]
            indexes = np.append(indexes, indexes)
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = self.__data_generation(list_IDs_temp)

        return X, y1, y2, y3, y4, y5, y6, y7

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _normalize(self, data, mode='max'):
        'Normalize waveforms in each batch'

        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert (max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data

        elif mode == 'std':
            std_data = np.std(data, axis=0, keepdims=True)
            assert (std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data

    def _scale_amplitude(self, data, rate):
        'Scale amplitude or waveforms'

        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2 * rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel(self, data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0):
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(self, data, rate):
        'Randomly replace values of one or two components to zeros in noise data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate:
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(self, data, rate):
        'Randomly add gaps (zeros) of different sizes into waveforms'

        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate:
            data[gap_start:gap_end, :] = 0
        return data

    def _add_noise(self, data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'

        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate:
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:, 0] + np.random.normal(0, np.random.uniform(0.01, 0.15) * max(data[:, 0]),
                                                             data.shape[0])
            data_noisy[:, 1] = data[:, 1] + np.random.normal(0, np.random.uniform(0.01, 0.15) * max(data[:, 1]),
                                                             data.shape[0])
            data_noisy[:, 2] = data[:, 2] + np.random.normal(0, np.random.uniform(0.01, 0.15) * max(data[:, 2]),
                                                             data.shape[0])
        else:
            data_noisy = data
        return data_noisy

    def _adjust_amplitude_for_multichannels(self, data):
        'Adjust the amplitude of multichaneel data'

        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert (tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
            data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(self, a=0, b=20, c=40):
        'Used for triangolar labeling'

        z = np.linspace(a, c, num=2 * (b - a) + 1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half] - a) / (b - a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c - z[second_half]) / (c - b)
        return y

    def _add_event(self, data, addp, adds, coda_end, snr, rate):
        'Add a scaled version of the event into the empty part of the trace'

        added = np.copy(data)
        additions = None
        spt_secondEV = None
        sst_secondEV = None
        coda_end_secondEV = None
        if addp and adds:
            s_p = adds - addp
            p_end = s_p + self.coda_ratio * s_p
            if (np.random.uniform(0, 1) < rate and all(snr >= 10.0) and
                    (data.shape[0] - s_p - 21 - (adds + self.coda_ratio * s_p)) > 20):
                secondEV_strt = np.random.randint(adds + self.coda_ratio * s_p + 21, data.shape[0] - s_p - 20)
                scaleAM = 1 / np.random.randint(1, 3)
                space = data.shape[0] - secondEV_strt
                added[secondEV_strt:secondEV_strt + space, 0] += data[addp:addp + space, 0] * scaleAM
                added[secondEV_strt:secondEV_strt + space, 1] += data[addp:addp + space, 1] * scaleAM
                added[secondEV_strt:secondEV_strt + space, 2] += data[addp:addp + space, 2] * scaleAM
                spt_secondEV = secondEV_strt
                if spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV + p_end + 21 <= data.shape[0]:
                    coda_end_secondEV = spt_secondEV + p_end
                if spt_secondEV and sst_secondEV and coda_end_secondEV:
                    additions = [spt_secondEV, sst_secondEV, coda_end_secondEV]
                    data = added
                elif spt_secondEV and sst_secondEV:
                    additions = [spt_secondEV, sst_secondEV]
                    data = added

        return data, additions

    def _shift_event(self, data, addp, adds, coda_end, snr, rate):
        'Randomly rotate the array to shift the event location'

        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]

            if addp + nrotate >= 0 and addp + nrotate < org_len:
                addp2 = addp + nrotate;
            else:
                addp2 = None;
            if adds + nrotate >= 0 and adds + nrotate < org_len:
                adds2 = adds + nrotate;
            else:
                adds2 = None;
            if coda_end + nrotate < org_len:
                coda_end2 = coda_end + nrotate
            else:
                coda_end2 = org_len
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end = coda_end2;
        return data, addp, adds, coda_end

    def _pre_emphasis(self, data, pre_emphasis=0.97):
        'apply the pre_emphasis'

        for ch in range(self.n_channels):
            bpf = data[:, ch]
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data


    def __data_generation(self, list_IDs_temp):
        'read the waveforms'
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        X_train1 = np.zeros((self.batch_size, self.dim, self.n_channels))
        X_train2 = np.zeros((self.batch_size, self.dim, self.n_channels))
        y1 = np.zeros((self.batch_size, self.dim, 1))
        y2 = np.zeros((self.batch_size, self.dim, 1))
        y3 = np.zeros((self.batch_size, self.dim, 1))
        y4 = np.zeros((self.batch_size, 1))
        y5 = np.zeros((self.batch_size, 1))
        y6 = np.zeros((self.batch_size, 1))
        y7 = np.zeros((self.batch_size, 2))
        # y8 = np.zeros((self.batch_size, 361))
        y8 = np.zeros((self.batch_size, 9))
        y9 = np.zeros((self.batch_size, 1))
        y10 = np.zeros((self.batch_size, 3))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            additions = None
            dataset = fl.get('data/' + str(ID))

            if ID.split('_')[-1] == 'EV':
                raw_data = np.array(dataset)
                trace_e = Trace(data=raw_data[:, 0])
                trace_e.stats.sampling_rate = 100
                trace_e.filter('bandpass', freqmin=1.0, freqmax=45, corners=2, zerophase=True)
                trace_n = Trace(data=raw_data[:, 1])
                trace_n.stats.sampling_rate = 100
                trace_n.filter('bandpass', freqmin=1.0, freqmax=45, corners=2, zerophase=True)
                trace_z = Trace(data=raw_data[:, 2])
                trace_z.stats.sampling_rate = 100
                trace_z.filter('bandpass', freqmin=1.0, freqmax=45, corners=2, zerophase=True)
                data_e = trace_e.data
                data_n = trace_n.data
                data_z = trace_z.data
                data = np.stack((data_e, data_n, data_z), axis=1)

                spt = int(dataset.attrs['p_arrival_sample']);
                sst = int(dataset.attrs['s_arrival_sample']);
                coda_end = int(dataset.attrs['coda_end_sample']);
                snr = dataset.attrs['snr_db'];
                distance = dataset.attrs['source_distance_km'];
                p_travel = dataset.attrs['p_travel_sec'];
                deep = dataset.attrs['source_depth_km'];
                azimuth_deg = dataset.attrs['back_azimuth_deg'];
                uncertainty = dataset.attrs['source_horizontal_uncertainty_km'];
                radians = np.deg2rad(azimuth_deg)
                y4[i, :] = distance
                y5[i, :] = p_travel
                y6[i, :] = deep
                y7[i, 0] = np.cos(radians)
                y7[i, 1] = np.sin(radians)
                y8[i, :] = deg2bins([azimuth_deg], 4, 9)[0, :]
                # y9[i, :] = uncertainty
                y10[i, :] = snr
                # y8[i, int(azimuth_deg)] = 1

            elif ID.split('_')[-1] == 'NO':
                data = np.array(dataset)

            ## augmentation
            if self.augmentation == True:
                if i <= self.batch_size // 2:
                    if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
                        data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr,
                                                                     self.shift_event_r / 2);
                    if self.norm_mode:
                        data = self._normalize(data, self.norm_mode)
                else:
                    if dataset.attrs['trace_category'] == 'earthquake_local':
                        if self.shift_event_r:
                            data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr,
                                                                         self.shift_event_r);

                        if self.add_event_r:
                            data, additions = self._add_event(data, spt, sst, coda_end, snr, self.add_event_r);

                        if self.add_noise_r:
                            data = self._add_noise(data, snr, self.add_noise_r);

                        if self.drop_channe_r:
                            data = self._drop_channel(data, snr, self.drop_channe_r);
                            data = self._adjust_amplitude_for_multichannels(data)

                        if self.scale_amplitude_r:
                            data = self._scale_amplitude(data, self.scale_amplitude_r);

                        if self.pre_emphasis:
                            data = self._pre_emphasis(data)

                        if self.norm_mode:
                            data = self._normalize(data, self.norm_mode)

                    elif dataset.attrs['trace_category'] == 'noise':
                        if self.drop_channe_r:
                            data = self._drop_channel_noise(data, self.drop_channe_r);

                        if self.add_gap_r:
                            data = self._add_gaps(data, self.add_gap_r)

                        if self.norm_mode:
                            data = self._normalize(data, self.norm_mode)

            elif self.augmentation == False:
                if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
                    data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r / 2);
                if self.norm_mode:
                    data = self._normalize(data, self.norm_mode)

            X[i, :, :] = data
        fl.close()

        return X, y1.astype('float32'), y2.astype('float32'), y3.astype('float32'), y4, y5, y6, y7, y8, y9, y10
