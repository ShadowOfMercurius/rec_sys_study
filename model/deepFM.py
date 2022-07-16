# author:zh

#deepFM wide&deep的变种，即将wide部分替换为FM模型

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten
from tensorflow.keras import Model
from collections import namedtuple
from wide_deep import deep_part
from FM_model import FM_model

SparseFeat = namedtuple('SparseFeat',['name','vocab_size','embed_dim'])

def DeepFM(fm_feature_dim,deep_feature,deep_units,fm_vdim,deep_output_dim=1):
    input_layer_dict={}
    # fm_feature也可以直接一次输入
    input_layer_dict['fm_input'] = Input(shape=(fm_feature_dim,))
    embed_layer_dict={}
    deep_input_list = []
    for feature in deep_feature:
        if isinstance(feature,SparseFeat):
            input_layer_dict[feature.name] = Input(shape=(1,))
            embed_layer_dict[feature.name] = Flatten()(Embedding(feature.vocab_size,feature.embed_dim)(input_layer_dict[feature.name]))
            deep_input_list.append(embed_layer_dict[feature.name])
        else:
            input_layer_dict[feature] = Input(shape=(1,))
            deep_input_list.append(input_layer_dict[feature])
    deep_layer = deep_part(deep_units, deep_output_dim)
    fm_layer = FM_model(fm_vdim)
    fm_output = fm_layer(input_layer_dict['fm_input'])
    deep_input = tf.concat(deep_input_list,axis=-1)
    deep_output = deep_layer(deep_input)
    output = tf.nn.sigmoid(0.5*(fm_output+deep_output))
    model = Model(input_layer_dict,output)
    return model




if __name__ == '__main__':
    '''
    与前面测试案例基本相同可以通过
    '''


