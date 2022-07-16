# author:zh

#ResNet残差网络  F(X)=H(X)-X，X为上一层输出，H(X)为该层输出，当X已经成熟时，任何对X的变动会使得loss变大，从而使得F(X)趋近于0，实现恒等映射

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten,Conv2D,BatchNormalization,Activation
from tensorflow.keras import Model
from collections import namedtuple
import pandas as pd


#标准resblock
class ResBlock(Layer):
    def __init__(self,filters,strides=1,residual_path=False):
        '''
        :param filters: 卷积核个数
        :param strides: strides为卷积时的步长
        :param residual_path: 决定是否需要进行下采样，当输入的大小和原大小不相同时，需要保证一致
        '''
        super(ResBlock,self).__init__()
        self.filters = filters()
        self.stride = strides
        self.residual_path = residual_path

        self.conv1 = Conv2D(filters,(3,3),strides=strides,padding='same',user_bias=False)
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters,(3,3),strides=1,padding='same',user_bias=False)
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')

        if residual_path:  #不带激活层，类似于线性变化
            self.down_cov = Conv2D(filters,(1,1),strides=strides,padding='same',user_bias=False)
            self.down_bn = BatchNormalization()

        self.act2 = Activation('relu')

    def call(self,input):
        residual = input
        x = self.c1(input)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)
        if self.residual_path:
            residual = self.down_b1(input)
            residual = self.down_bn(residual)
        output = self.a2(y+residual)
        return output


class ResNet18(Model):
    def __init__(self,block_list,initial_filters=64):
        '''
        :param block_list: 每层block个数
        :param initial_filters:
        '''
        super(ResNet18,self).__init__()
        self.num_blocks=len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1=Conv2D(self.out_filters,(3,3),strides=1,padding='same',use_bias=False)
        self.b1=BatchNormalization()
        self.a1=Activation('relu')
        self.blocks = tf.keras.model.Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id !=0 and layer_id == 0:
                    block =ResBlock(self.out_filters,strides=2,residual_path=True)
                else:
                    block =ResBlock(self.out_filters,residual_path=False)
                self.blocks.add(block)
            self.out_filters *= 2
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()   #[b,h,w,c]->[b,c]
        self.f1 = tf.keras.layers.Dense(10,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())

    def call(self,inputs):
        x=self.c1(inputs)
        x=self.b1(x)
        x=self.a1(x)
        x=self.blocks(x)
        x=self.p1(x)
        y=self.f1(x)
        return y




