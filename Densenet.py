# !/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras import layers, optimizers, Input, regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential, load_model,Model
from tensorflow.keras.layers import Reshape, BatchNormalization, Lambda
from tensorflow.keras.layers import LayerNormalization, Activation
import matplotlib.pyplot as plt
from sklearn import manifold

def build_model_withDensnet():
    input = Input(shape=(28, 1000, 1), name='partial_train_Datain')
    # input_4_8 = Input(shape=(28, 1000, 1), name='partial_train_Datain_4_8')
    input_8_13 = Input(shape=(28, 1000, 1), name='partial_train_Datain_8_13')
    input_13_32 = Input(shape=(28, 1000, 1), name='partial_train_Datain_13_32')
    # input_1_4 = Input(shape=(28, 1000, 1), name='partial_train_Datain_1_4')
    # input_raw = Input(shape=(20, 1000, 1), name='partial_train_Datain')
    b = Densnet(input)
    # c = Densnet(input_4_8)
    d = Densnet(input_8_13)
    e = Densnet(input_13_32)
    # f = Densnet(input_1_4)

    b = layers.Flatten(name='raw')(b)
    # c = layers.Flatten(name='theta')(c)
    d = layers.Flatten(name='alpha')(d)
    e = layers.Flatten(name='beta')(e)
    # f = layers.Flatten(name='delta')(f)
    #
    a = layers.concatenate([b, d, e], name='FC1')
    #
    output = layers.Dense(2, kernel_constraint=max_norm(0.25), name='dense')(a)
    output = Activation('softmax', name='FC3')(output)
    # output = layers.Dense(2, activation='softmax', kernel_constraint=max_norm(0.25))(c)
    # output = backend.clip(output, 1e-10, 1.0)
    #
    model = Model(inputs = [input,input_8_13,input_13_32], outputs = output)
    # model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
    return model
def build_model_withDensnet2():
    input = Input(shape=(20, 1000, 1), name='partial_train_Datain')
    b = Densnet2(input)
    #
    output = layers.Dense(2, kernel_constraint=max_norm(0.25), name='dense')(b)
    output = Activation('softmax', name='FC3')(output)
    #
    model = Model(inputs=input, outputs=output)
    # model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
    return model
def DenseLayer(x, nb_filter, bn_size=4, drop_rate=0.5, weight_decay = 1e-4, cnn_size = 16):
    # Bottleneck layers
    # x = BatchNormalization(axis=3)(x)
    # x = Activation('elu')(x)
    # x = layers.Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), use_bias=False,
    #                   padding='same',kernel_regularizer = regularizers.l2(weight_decay))(x)

    # Composite function
    N = cnn_size
    x = layers.Conv2D(nb_filter, (1, N), strides=(1, 1), padding='same', use_bias=False
                      ,kernel_regularizer = regularizers.l2(weight_decay), kernel_constraint=max_norm(1.))(x)
    x = Activation('elu')(x)
    x = BatchNormalization(axis=3)(x)
    if drop_rate: x = layers.Dropout(drop_rate)(x)

    return x

def DenseBlock(x, nb_layers, growth_rate, drop_rate, cnn_size):
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate, weight_decay = 1e-4, cnn_size = cnn_size )
        x = layers.concatenate([x, conv], axis=3)
    return x

def TransitionLayer(x, compression=0.5, is_max=0, weight_decay = 1e-4):
    # nb_filter = int(x.shape.as_list()[-1] * compression)
    # x = BatchNormalization(axis=3)(x)
    # x = Activation('elu')(x)
    # x = layers.Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same', use_bias=False
    #                   ,kernel_regularizer = regularizers.l2(weight_decay))(x)
    if is_max != 0:
        x = layers.MaxPooling2D(pool_size=(1, 5), strides=(1, 5))(x)
    else:
        x = layers.AveragePooling2D(pool_size=(1, 5), strides=(1, 5))(x)

    return x

def SEBlock(se_ratio, activation = "elu", data_format = 'channels_last', ki = "he_normal"):
    '''
    se_ratio : ratio for reduce the filter number of first Dense layer(fc layer) in block
    activation : activation function that of first dense layer
    data_format : channel axis is at the first of dimension or the last
    ki : kernel initializer
    '''
    def f(input_x):
        channel_axis = -1 if data_format == 'channels_last' else 1
        input_channels = input_x.shape[channel_axis]
        reduced_channels = input_channels // se_ratio
        #Squeeze operation
        x = layers.GlobalAveragePooling2D()(input_x)
        x = layers.Reshape(1,1,input_channels)(x) if data_format == 'channels_first' else x
        x = layers.Dense(reduced_channels, kernel_initializer= ki,kernel_constraint=max_norm(2.))(x)
        x = layers.Activation(activation)(x)
        #Excitation operation
        x = layers.Dense(input_channels, kernel_initializer=ki, kernel_constraint=max_norm(2.),name = 'channel_weight1')(x)
        x = layers.Activation('sigmoid',name = 'channel_weight2')(x)
        x = layers.Permute(dims=(3,1,2))(x) if data_format == 'channels_first' else x
        x = layers.multiply([input_x, x])
        return x
    return f

def Densnet(input_tensor):
    c = layers.Conv2D(filters=40, kernel_size=(1, 64), use_bias=False, padding='same')(input_tensor)
    c = layers.BatchNormalization()(c)
    c = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(28, 1), use_bias=False,
                               depthwise_constraint=max_norm(1.))(c)
    c = Activation('elu')(c)
    c = layers.BatchNormalization()(c)
    c = DenseBlock(c, nb_layers = 2, growth_rate = 7, drop_rate = 0.5, cnn_size = 64)
    c = layers.BatchNormalization()(c)
    c = layers.AveragePooling2D(pool_size=(1, 5), strides=(1, 5))(c)
    # c = layers.Conv2D(filters=20,kernel_size=(1, 1), use_bias=False, padding='same',
    #                   kernel_regularizer=regularizers.l2(1e-4), kernel_constraint=max_norm(2.))(c)
    # c = Activation('elu')(c)
    # c = layers.BatchNormalization()(c)
    c = DenseBlock(c, nb_layers = 2, growth_rate=7, drop_rate=0.5, cnn_size = 20)
    c = layers.BatchNormalization()(c)
    c = layers.AveragePooling2D(pool_size=(1, 5), strides=(1, 5))(c)
    c = layers.Conv2D(filters=20,kernel_size=(1, 1), use_bias=False, padding='same',kernel_constraint=max_norm(2.))(c)
    c = Activation('elu')(c)
    c = layers.BatchNormalization()(c)
    c = layers.AveragePooling2D(pool_size=(1, 2), strides=(1, 2))(c)
    output = layers.Flatten()(c)
    return output

def Densnet2(input_tensor):
    c = layers.Conv2D(filters=40, kernel_size=(1, 64), use_bias=False, padding='same')(input_tensor)
    c = layers.BatchNormalization(name='tem')(c)
    c = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(20, 1), use_bias=False,
                               depthwise_constraint=max_norm(1.))(c)
    c = Activation('elu')(c)
    c = layers.BatchNormalization(name='spc')(c)
    c = DenseBlock(c, nb_layers = 2, growth_rate = 5, drop_rate = 0.5, cnn_size = 64)
    c = layers.BatchNormalization(name='denseblock1')(c)
    c = layers.AveragePooling2D(pool_size=(1, 5), strides=(1, 5))(c)
    # c = layers.Conv2D(filters=20,kernel_size=(1, 1), use_bias=False, padding='same',
    #                   kernel_regularizer=regularizers.l2(1e-4), kernel_constraint=max_norm(2.))(c)
    # c = Activation('elu')(c)
    # c = layers.BatchNormalization()(c)
    c = DenseBlock(c, nb_layers = 2, growth_rate=5, drop_rate=0.5, cnn_size = 20)
    c = layers.BatchNormalization(name='denseblock2')(c)
    c = layers.AveragePooling2D(pool_size=(1, 5), strides=(1, 5))(c)
    c = layers.Conv2D(filters=20,kernel_size=(1, 1), use_bias=False, padding='same',kernel_constraint=max_norm(2.))(c)
    c = Activation('elu')(c)
    c = layers.BatchNormalization(name='1CNN')(c)
    c = layers.AveragePooling2D(pool_size=(1, 5), strides=(1, 5))(c)
    output = layers.Flatten()(c)
    return output

def draw_t_SNE(out, label):
    color = label
    color = [np.argmax(i) for i in color]  # 将one-hot编码转换为整数
    color = np.stack(color, axis=0)
    n_neighbors = 2  # 一共有多少个类别
    n_components = 2  # 降维成几维 2或者3
    # out = out.reshape(160,20000)
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    # 训练模型
    out = out.reshape(160,-1)
    y = ts.fit_transform(out)
    cm = 'bwr'  # 调整颜色
    # cm = colormap()
    plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
    # 显示图像
    plt.show()

