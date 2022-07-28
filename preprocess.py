# _*_ coding: utf-8 _*_

import os
import numpy as np
import scipy
from scipy.fftpack import fft,fftshift
from sklearn import preprocessing
import matplotlib.pyplot as plt

def data_standardization2(data):  # Z-Score标准化/
    mean = data.mean(axis=2)
    x = np.size(data, 0)
    y = np.size(data, 1)
    z = np.size(data, 2)
    for k in range(x):
        for i in range(y):
            for j in range(z):
                data[k, i, j] -= mean[k, i]
            std = data[k, i, :].std(axis=0)
            data[k, i, :] /= std
    return data
def z_score(data):  # Z-Score标准化/
    data -= np.mean(data)
    std = np.std(data)
    data = data /  std
    return data
def data_standardization(data):  # max-min标准化
    x = np.size(data, 0)
    y = np.size(data, 1)
    z = np.size(data, 2)
    for k in range(x):
        for i in range(y):
            Max = max(data[k, i, :])
            Min = min(data[k, i, :])
            Range = Max - Min
            for j in range(z):
                data[k, i, j] = (data[k, i, j] - Min) / Range
            # dataMax = max(data[k, i, :])
            # dataMin = min(data[k, i, :])
    return data
def detrend(data):  #
    DATA = data.copy()
    x = np.size(DATA, 0)
    y = np.size(DATA, 1)
    z = np.size(DATA, 2)
    for k in range(x):
        for i in range(y):
            datatemp = DATA[k, i, :]
            DATA[k, i, :] = scipy.signal.detrend(datatemp)
    return DATA
def down_sample(data, original_sample,aim_sample,type):
    step = int(original_sample/aim_sample)
    epochs = int(data.shape[2]/step)
    trials = data.shape[0]
    Channels = data.shape[1]
    New_data = np.zeros((trials, Channels, epochs))
    for i in range(trials):
        for j in range(epochs):
            if type == 1:
                New_data[i,:,j] = np.mean(data[i,:,j*step:(j+1)*step], 1)
            if type == 2:
                New_data[i,:,j] = np.max(data[i, :, j * step:(j + 1) * step], 1)
            if type == 3:
                New_data[i,:,j] = data[i, :, j * step]
    return New_data
def data_crop(data,class_in,step,win_num):
    Event1, Event2, Event3 = [], [], []
    class_in = np.squeeze(class_in).T
    n_crop = int((data.shape[2]-win_num)/step)
    for index, lable in enumerate(class_in):
        if lable in [1, 2]:
            Event3.append(int(lable))
            for i in range(n_crop):
                Event1.append([int(lable - 1)])
    Event1 = np.squeeze(Event1)
    Event1 = np.expand_dims(Event1, axis=1)
    Data = np.zeros((data.shape[0]*n_crop,data.shape[1],win_num))
    for i in range(data.shape[0]):
        for j in range(n_crop):
            Data[i*n_crop+j,:,:] = data[i,:,j*step:win_num+j*step]
    print("end")
    return Data,Event1
def draw_fft(data, sample_freq,N ):
    fft_data = fft(data)
    # 这里幅值要进行一定的处理，才能得到与真实的信号幅值相对应
    fft_amp0 = np.array(np.abs(fft_data) / N * 2)  # 用于计算双边谱
    direct = fft_amp0[0]
    fft_amp0[0] = 0.5 * direct
    N_2 = int(N / 2)

    fft_amp1 = fft_amp0[0:N_2]  # 单边谱
    fft_amp0_shift = fftshift(fft_amp0)  # 使用fftshift将信号的零频移动到中间

    # 计算频谱的频率轴
    list0 = np.array(range(0, N))
    list1 = np.array(range(0, int(N / 2)))
    list0_shift = np.array(range(0, N))
    freq0 = sample_freq * list0 / N  # 双边谱的频率轴
    freq1 = sample_freq * list1 / N  # 单边谱的频率轴
    freq0_shift = sample_freq * list0_shift / N - sample_freq / 2  # 零频移动后的频率轴

    # 绘制结果
    plt.figure()
    # 双边谱
    # plt.subplot(221)
    # plt.plot(freq0, fft_amp0)
    # plt.title(' spectrum two-sided')
    # plt.ylim(0, 6)
    # plt.xlabel('frequency  (Hz)')
    # plt.ylabel(' Amplitude ')
    # 单边谱
    plt.subplot(222)
    plt.plot(freq1, fft_amp1)
    plt.title(' spectrum single-sided')
    plt.ylim(0, 6)
    plt.xlabel('frequency  (Hz)')
    plt.ylabel(' Amplitude ')
    # 移动零频后的双边谱
    # plt.subplot(223)
    # plt.plot(freq0_shift, fft_amp0_shift)
    # plt.title(' spectrum two-sided shifted')
    # plt.xlabel('frequency  (Hz)')
    # plt.ylabel(' Amplitude ')
    # plt.ylim(0, 6)
    plt.show()
def filter(DATA, n_trials, n_channels,low,high,frequency):
    b, a = scipy.signal.butter(3, [low / (frequency/2), high / (frequency/2)], btype='bandpass')
    # 滤波和标准化(Z-SCORE)
    result = DATA.copy()
    result = result.squeeze()
    for ss in range(n_trials):
        for yy in range(n_channels):
            data_temp1 = result[ss, yy, :]
            data_temp1 = scipy.signal.filtfilt(b, a, data_temp1)
            # data_temp1 = preprocessing.scale(data_temp1)
            result[ss,yy,:] = data_temp1
    result = np.expand_dims(result, axis=3)
    return result
def drawplot(all_loss, all_val_loss, all_acc, all_val_acc,name):
    # all_loss = [np.mean([x[i] for x in all_loss]) for i in range(50)]
    # all_val_loss = [np.mean([x[i] for x in all_val_loss]) for i in range(50)]
    # all_acc = [np.mean([x[i] for x in all_acc]) for i in range(50)]
    # all_val_acc = [np.mean([x[i] for x in all_val_acc]) for i in range(50)]
    # "bo" is for "blue dot"
    plt.plot(range(1, len(all_loss) + 1), all_loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(range(1, len(all_val_loss) + 1), all_val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    name1 = name + '-loss.png'
    plt.savefig(name1)
    plt.cla()
    # plt.show()

    # "bo" is for "blue dot"
    plt.plot(range(1, len(all_acc) + 1), all_acc, 'bo', label='Training acc')
    # b is for "solid blue line"
    plt.plot(range(1, len(all_val_acc) + 1), all_val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    name2 = name + '-acc.png'
    plt.savefig(name2)
    plt.cla()
    # plt.show()
    plt.close()


'''
# ch_names = ["FPz", "FP1", "FP2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2",
#             "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3", "FC4",
#             "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4", "C5", "C6",
#             "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
#             "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3", "PO4", "PO5",
#             "PO6", "PO7", "PO8", "Oz", "O1", "O2"]
# ch_str_name = "FPz,FP1,FP2,AF3,AF4,AF7,AF8,Fz,F1,F2,F3,F4,F5,F6,F7,F8," \
#               "FCz,FC1,FC2,FC3,FC4,FC5,FC6,FT7,FT8,Cz,C1,C2,C3,C4,C5,C6," \
#               "T7,T8,CP1,CP2,CP3,CP4,CP5,CP6,TP7,TP8,Pz,P3,P4,P5,P6,P7," \
#               "P8,POz,PO3,PO4,PO5,PO6,PO7,PO8,Oz,O1,O2"

# 调整滤波参数，暂时不用
    # filtAllowance = 2
    # bandFiltCutF = (4, 40)
    # nFreq = aim_samplingrate / 2
    # aStop = 30  # stopband attenuation
    # aPass = 3
    # fPass = (np.array(bandFiltCutF) / nFreq).tolist()
    # fStop = [(bandFiltCutF[0] - filtAllowance) / nFreq, (bandFiltCutF[1] + filtAllowance) / nFreq]
    # find the order
    # [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
    # b, a = signal.cheby2(N, aStop, fStop, 'bandpass')
    # b, a = signal.cheby2(9, aStop, fStop, 'bandpass')
    # b, a = signal.butter(3, [1 / 125, 40 / 125], btype='bandpass')
    #
'''


