# author:zh

##NFM(Neural Factorization Machines):对FM利用深度学习的改进，将FM的二阶交叉部分用神经网络来代替，将特征embedding后呈上原始值，送入deep部分

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten
from tensorflow.keras import Model
from collections import namedtuple
from wide_deep import wide_part,deep_part
from utils import SparseFeat


def NFM(lr_input_dim,bi_layer_input_feature,hidden_units,output_dim=1,activatio='relu'):
    input_layer_dict={}
    embed_layer_dict = {}
    bi_input_list = []
    input_layer_dict['lr_input'] = Input(shape=(lr_input_dim,))
    for feature in bi_layer_input_feature:
        input_layer_dict[feature.name] = Input(shape=(1,))
        embed_layer_dict[feature.name] = Flatten()(Embedding(feature.vocab_size,feature.embed_dim)(input_layer_dict[feature.name]))
        bi_input_list.append(embed_layer_dict[feature.name])      #list,共feature个元素，每个元素[batch_size,embed_dim]
    bi_input1 = tf.transpose(tf.convert_to_tensor(bi_input_list),[1,0,2])
    inter_part1 = tf.pow(tf.reduce_sum(bi_input1,axis=1),2)
    inter_part2 = tf.reduce_sum(tf.pow(bi_input1,2),axis=1)
    bi_input2 = 0.5*(inter_part1-inter_part2)
    wide_layer = wide_part()
    deep_layer = deep_part(hidden_units,output_dim,activation='relu')
    wide_output = wide_layer(input_layer_dict['lr_input'])
    deep_output = deep_layer(bi_input2)
    output = tf.nn.sigmoid(0.5*(wide_output+deep_output))
    model = Model(input_layer_dict,output)
    return model


if __name__ == '__main__':
    feature_columns2 = [
        SparseFeat('f21', 5, 8),
        SparseFeat('f31', 5, 8),
    ]
    nfm = NFM(4, feature_columns2, [20, 20])
    nfm.summary()
