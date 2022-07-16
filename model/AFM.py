# author:zh

#AFM(attention Factorization Machines):对于在二阶交叉层在进行pooling前进行attention机制(该网络机制为单层MLP），然后利用加权平均

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten
from tensorflow.keras import Model
from collections import namedtuple
from wide_deep import wide_part,deep_part
from utils import SparseFeat

class Attention_layer(Layer):
    def __init__(self):
        super(Attention_layer,self).__init__()

    def build(self,input_shape):
        self.w_matrix = Dense(input_shape[1],activation='relu')
        self.h_matrix = Dense(1,activation=None)

    def call(self,inputs):
        #inputs:[batch_size,交叉后特征个数，embed_dim】
        x = self.w_matrix(inputs)  #[batch_size,交叉特征个个数，交叉特征个数】
        x = self.h_matrix(x)       #[batch_size,交叉特征个数，1]
        attention_score = tf.transpose(tf.nn.sigmoid(x),[0,2,1])
        attention_output = tf.matmul(attention_score,inputs)
        output = tf.reshape(attention_output,shape=(-1,inputs.shape[2]))
        return output


def get_bi_vectors(embed_list):
    vec_nums = len(embed_list)
    bi_list = []
    for i in range(vec_nums):
        for j in range(i+1,vec_nums):
            bi_list.append(tf.multiply(embed_list[i],embed_list[j]))
    bi_vec_list = tf.transpose(tf.convert_to_tensor(bi_list),[1,0,2])
    return bi_vec_list


def AFM(lr_input_dim,bi_layer_input_feature):
    input_layer_dict = {}
    embed_layer_dict = {}
    bi_input_list = []
    input_layer_dict['lr_input'] = Input(shape=(lr_input_dim,))
    for feature in bi_layer_input_feature:
        input_layer_dict[feature.name] = Input(shape=(1,))
        embed_layer_dict[feature.name] = Flatten()(
            Embedding(feature.vocab_size, feature.embed_dim)(input_layer_dict[feature.name]))
        bi_input_list.append(embed_layer_dict[feature.name])
    bi_vec_list = get_bi_vectors(bi_input_list)
    wide_layer = wide_part()
    attention_part = Attention_layer()
    wide_output = wide_layer(input_layer_dict['lr_input'])
    attention_output = attention_part(bi_vec_list)
    attention_output2 = Dense(1,activation=None)(attention_output)
    output = tf.nn.sigmoid(0.5*(wide_output+attention_output2))
    model = Model(input_layer_dict,output)
    return model

if __name__ == '__main__':
    feature_columns3 = [
        SparseFeat('f21', 5, 8),
        SparseFeat('f31', 5, 8),
        SparseFeat('f41', 5, 8),
        SparseFeat('f51', 5, 8),
    ]
    nfm = AFM(4, feature_columns3)
    nfm.summary()
