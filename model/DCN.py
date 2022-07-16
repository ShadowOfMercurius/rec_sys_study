# author:zh

#DCN(deep crossing network),将wide&deep部分替换成cross模型，从而避免了wide部分的人工特征交叉，同时学习低维特征和高维特征

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten
from tensorflow.keras import Model
from collections import namedtuple
from wide_deep import deep_part
from utils import build_input_dict_for_widedeep,SparseFeat



#SparseFeat = namedtuple('SparseFeat',['name','vocab_size','embed_dim'])

class Cross_part(Layer):
    def __init__(self,layer_cnts):
        super(Cross_part,self).__init__()
        self.layer_cnts = layer_cnts

    def build(self,input_shape):
        self.w_list=[]
        self.b_list=[]
        for idx in range(self.layer_cnts):
            w = self.add_weight(name='w_'+str(idx),shape=(input_shape[-1],1),initializer=tf.random.normal_initializer(),
                              trainable=True,regularizer=tf.keras.regularizers.l1(1e-4))
            b = self.add_weight(name='b_'+str(idx),shape=(input_shape[-1],1),initializer=tf.zeros_initializer(),
                              trainable=True)
            self.w_list.append(w)
            self.b_list.append(b)

    def call(self,input):
        origin = tf.expand_dims(input,axis=2)   #[batch_size,input_shape]->[batch_size,input_shape,1]
        x = origin
        for idx in range(self.layer_cnts):
            cur_w = self.w_list[idx]
            cur_b = self.b_list[idx]
            tmp = tf.matmul(tf.transpose(x,[0,2,1]),cur_w)
            x = tf.matmul(origin,tmp)+x+cur_b
        x = tf.squeeze(x,axis=2)
        return x

def DCN_model(cross_input_shape,deep_feature,cross_layer_cnts,hidden_units,dnn_output_dim,activation='relu'):
    cross_input,dnn_input,input_layer_dict = build_input_dict_for_widedeep(cross_input_shape,deep_feature,'cross_input')
    cross_part = Cross_part(cross_layer_cnts)
    dnn_part = deep_part(hidden_units,dnn_output_dim,activation)
    cross_output = cross_part(cross_input)
    dnn_output = dnn_part(dnn_input)
    last_input = tf.concat([dnn_output,cross_output],axis=-1)
    output = Dense(1,activation='sigmoid')(last_input)
    model = Model(input_layer_dict,output)
    return model



if __name__ == '__main__':
    feature_columns = ['f11',
                       SparseFeat('f21', 5, 8),
                       SparseFeat('f31', 5, 8),
                       ]
    dcn = DCN_model(4, feature_columns, 3, [20, 20], 3)
    dcn.summary()
    #.....





