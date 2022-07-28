# _*_ coding: utf-8 _*_

# import mne
import os, preprocess, Densenet, EEGModels
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import scipy.io as sciio
import tensorflow as tf
from scipy.stats import ranksums
from tensorflow.keras import Sequential, layers, optimizers, callbacks
from tensorflow.keras import Model
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import random

if __name__ == '__main__':
    random.seed(7)
    np.random.seed(7)
    tf.random.set_seed(7)
    # 限制GPU资源使用
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    #
    # data information
    subject_num = 1
    raw_samplingrate = 1000
    aim_samplingrate = 250
    chans = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20] #20
    chans2 = [0, 56, 57, 54, 2, 3, 7, 32, 8, 45, 44, 11, 34, 12, 35, 46, 47, 17, 16, 38, 18, 41, 23, 22, 48, 60, 28, 27]#28
    chans3 = [1, 59, 58, 55, 6, 5, 49, 9, 10, 33, 50, 36, 14, 37, 15, 51, 19, 40, 20, 52, 21, 53, 42, 25, 26, 61, 30, 31]#28
    model_name = []
    ACC_RATE, Confusion_matrix = np.zeros((54)), np.zeros((54, 5))
    for i in range(54):
        model_name.append(0)
        model_name[i] = r"C:\Users\zhang\Desktop\新建文件夹 2-1\Densenet-hemisphere-right\S" + str(i + 1)
        isExists = os.path.exists(model_name[i])
        if not isExists:
            os.makedirs(model_name[i])
    for subject_num in range(13, 31):
        raw_train = sciio.loadmat('G:\EEGnetcode-KRdatasets-Jiayang Zhang\Dataset\Traindata' + str(subject_num) + '.mat')['smt']
        raw_test = sciio.loadmat('G:\EEGnetcode-KRdatasets-Jiayang Zhang\Dataset\Testdata' + str(subject_num) + '.mat')['smt']
        train_class = sciio.loadmat('G:\EEGnetcode-KRdatasets-Jiayang Zhang\Dataset\Trainclass' + str(subject_num) + '.mat')['class']
        test_class = sciio.loadmat('G:\EEGnetcode-KRdatasets-Jiayang Zhang\Dataset\Testclass' + str(subject_num) + '.mat')['class']
        train_data = raw_train.transpose(1, 2, 0)
        test_data = raw_test.transpose(1, 2, 0)
        # 降采样
        train_data = preprocess.down_sample(train_data, raw_samplingrate, aim_samplingrate, type=3)
        test_data = preprocess.down_sample(test_data, raw_samplingrate, aim_samplingrate, type=3)
        del raw_train, raw_test  # 节约内存
        # 数据整合
        DATA = np.concatenate((train_data, test_data), axis=0)
        DATA = DATA[:, np.array(chans3), :].squeeze()
        CLASS = np.concatenate((train_class.T, test_class.T), axis=0)
        class_acc = CLASS
        del train_data, test_data, train_class, test_class  # 节约内存
        n_trials, n_channels, timepoints = DATA.shape[0], DATA.shape[1], DATA.shape[2]
        # train_Datain, train_class = preprocess.data_crop(train_Datain, train_class, step=10, win_num=250)
        DATA = preprocess.detrend(DATA) #去基线
        class_temp = class_acc.squeeze() #左手标签是2，右手为1(KR dataset 2019)
        LEFT = [i for i, x in enumerate(class_temp) if x == 2]
        RIGHT = [i for i, x in enumerate(class_temp) if x == 1]
        LEFT_DATA, RIGHT_DATA = DATA[np.array(LEFT),: , :], DATA[np.array(RIGHT),: , :]
        #将data和class按左右label各一个重新排序，确保训练验证和测试集的标签平衡
        index_l, index_r = 0, 0
        for index in range (n_trials):
            if index % 2 == 0:
                DATA[index,:,:] = LEFT_DATA[index_l]
                class_acc[index,:], CLASS[index,:] = 2, 2
                index_l = index_l + 1
            else:
                DATA[index,:,:] = RIGHT_DATA[index_r]
                class_acc[index, :], CLASS[index, :] = 1, 1
                index_r = index_r + 1
        # DATA = preprocess.filter(DATA, n_trials, n_channels, 4, 32, aim_samplingrate)#滤波
        CLASS = tf.keras.utils.to_categorical(CLASS - 1, num_classes=2, dtype="float32")
        # CNN卷积需要增加维度
        DATA = np.expand_dims(DATA, axis=3)
        # #
        callback = callbacks.EarlyStopping(monitor='val_loss', patience=200, mode="min",
                                           restore_best_weights=True, verbose=2)
        k = 10
        acc_n = 0
        num_val_samples = len(DATA) // k
        all_val_loss, all_loss = 0, 0
        all_val_acc, all_acc = 0, 0
        TN, FP, FN, TP, KV = 0, 0, 0, 0, 0
        #
        # Fit_n1, Fit_n2, Fit_n3 = [], [], []
        current_acc = np.zeros((num_val_samples))
        for i in range(k):
            current_acc_n = 0
            print('processing fold #', i)
            val_Datain = DATA[i * num_val_samples: (i + 1) * num_val_samples]
            val_Class = CLASS[i * num_val_samples: (i + 1) * num_val_samples]
            test_Datain = DATA[(i + 1) * num_val_samples: (i + 2) * num_val_samples]
            test_Class = CLASS[(i + 1) * num_val_samples: (i + 2) * num_val_samples]
            acc_class = class_acc[(i + 1) * num_val_samples: (i + 2) * num_val_samples]
            partial_train_Datain = np.concatenate([DATA[:i * num_val_samples],
                                                   DATA[(i + 2) * num_val_samples:]], axis=0)
            partial_train_Classin = np.concatenate([CLASS[:i * num_val_samples],
                                                    CLASS[(i + 2) * num_val_samples:]], axis=0)
            if i == 9:
                test_Datain = DATA[: num_val_samples]
                test_Class = CLASS[: num_val_samples]
                acc_class = class_acc[: num_val_samples]
                partial_train_Datain = DATA[num_val_samples: i * num_val_samples]
                partial_train_Classin = CLASS[num_val_samples: i * num_val_samples]

            partial_train_Classin = partial_train_Classin.squeeze()
            val_Class = val_Class.squeeze()

            # Densenet
            # partial_train_Datain_4_8 = preprocess.filter(partial_train_Datain, 160, 28, 4, 8, 250)
            partial_train_Datain_8_13 = preprocess.filter(partial_train_Datain, 160, 28, 7, 13, 250)
            partial_train_Datain_13_32 = preprocess.filter(partial_train_Datain, 160, 28, 12, 32, 250)
            # partial_train_Datain_1_4 = preprocess.filter(partial_train_Datain, 160, 28, 1, 5, 250)
            # val_Datain_4_8 = preprocess.filter(val_Datain, 20, 28, 4, 8, 250)
            val_Datain_8_13 = preprocess.filter(val_Datain, 20, 28, 7, 13, 250)
            val_Datain_13_32 = preprocess.filter(val_Datain, 20, 28, 12, 32, 250)
            # val_Datain_1_4 = preprocess.filter(val_Datain, 20, 28, 1, 5, 250)
            # test_Datain_4_8 = preprocess.filter(test_Datain, 20, 28, 4, 8, 250)
            test_Datain_8_13 = preprocess.filter(test_Datain, 20, 28, 7, 13, 250)
            test_Datain_13_32 = preprocess.filter(test_Datain, 20, 28, 12, 32, 250)
            # test_Datain_1_4 = preprocess.filter(test_Datain, 20, 28, 1, 5, 250)
            model = Densenet.build_model_withDensnet()
            # shuffle keras默认开启
            history = model.fit([partial_train_Datain, partial_train_Datain_8_13
                                    , partial_train_Datain_13_32],
                                partial_train_Classin, epochs=1000, batch_size=16,
                                validation_data=([val_Datain, val_Datain_8_13,
                                                  val_Datain_13_32], val_Class),
                                callbacks=[callback], verbose=2)
            Result = np.argmax(model.predict([test_Datain, test_Datain_8_13,
                                              test_Datain_13_32]), axis=-1)

            # model = Densenet.build_model_withDensnet2()
            # model = EEGModels.ShallowConvNet(nb_classes=2)
            # partial_train_Datain = partial_train_Datain.transpose(0, 3, 1, 2)
            # val_Datain = val_Datain.transpose(0, 3, 1, 2)
            # test_Datain = test_Datain.transpose(0, 3, 1, 2)
            # history = model.fit(partial_train_Datain, partial_train_Classin, epochs=1000, batch_size=16,
            #                     validation_data=(val_Datain, val_Class),
            #                     callbacks=[callback], verbose=2)
            # Result = np.argmax(model.predict(test_Datain), axis=-1)
            #
            for index_R in range(num_val_samples):
                if Result[index_R] + 1 == acc_class[index_R]:
                    current_acc_n = current_acc_n + 1
                    acc_n = acc_n + 1
            current_acc[i] = current_acc_n / num_val_samples
            name_CUR = model_name[subject_num - 1] + '\\S' + str(subject_num) + '-CURACC' + '.txt'
            np.savetxt(name_CUR, current_acc, fmt='%f', delimiter=',  ')
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            name = model_name[subject_num - 1] + '\\S' + str(subject_num) + '-' + str(i + 1) + '.txt'
            name2 = model_name[subject_num - 1] + '\\S' + str(subject_num) + '-' + str(i + 1)
            preprocess.drawplot(loss, val_loss, acc, val_acc, name2)
            #画t-SNE
            # dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
            # out = dense1_layer_model.predict([partial_train_Datain])
            # Densenet.draw_t_SNE(out, partial_train_Classin)
            # Densenet.draw_t_SNE(out, tf.keras.utils.to_categorical(acc_class - 1, num_classes=2, dtype="float32"))
            # confusion matrix
            # tn, fp, fn, tp = confusion_matrix(acc_class-1, Result).ravel()
            # kappa_value = cohen_kappa_score(acc_class-1, Result)
            # TN, FP, FN, TP, KV = TN + tn, FP + fp, FN + fn, TP + tp, KV + kappa_value
            # parameters = np.array([loss, val_loss, acc, val_acc])
            # parameters = parameters.transpose((1, 0))
            # p = np.zeros((parameters.shape[0], 5))
            # for j in range(p.shape[0]):
            #     p[j, 0] = j
            #     p[j,1:5] = parameters[j, :]

            # name1 = model_name[subject_num - 1] + '\\S' + str(subject_num) + 'M' + str(i + 1) + '.h5'
            # model.save(name1)
            ## 删除内存
            # del val_Datain_4_13, val_Datain_13_32, test_Datain_4_13, test_Datain_13_32
        ACC_RATE[subject_num - 1] = acc_n / len(DATA)
        name = model_name[subject_num - 1] + '\\S' + str(subject_num) + '-ACC' + '.txt'
        np.savetxt(name, ACC_RATE, fmt='%f', delimiter=',  ')
        # Confusion_matrix = np.array([TN, FP, FN, TP, KV / 10])
        # name = model_name[subject_num - 1] + '\\S' + str(subject_num) + '-CM' + '.txt'
        # np.savetxt(name, Confusion_matrix, fmt='%f', delimiter=',  ')
print('断点')
