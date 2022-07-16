# author:zh

#DIEN（deep interest Evolution Network）
#DIN和其他模型中直接用行为代表兴趣，不能表现用户的隐藏兴趣，且没关注到用户的兴趣变化
#其他论文中提到使用RNN来提取兴趣但效果不佳，因此可能需要特殊的结构来提取（跟点击序列有关（a\b\c和b\a\c的差别远小于nlp）

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten,BatchNormalization,PReLU,GRU
from tensorflow.keras import Model
from collections import namedtuple
from wide_deep import wide_part,deep_part
from utils import SparseFeat,DenseFeat,behaviorFeat
from DIN import Attention_part,Dice,DeepPart


class GRU_GATES(Layer):
    def __init__(self,units):
        super(GRU_GATES,self).__init__()
        self.linear_part0 = Dense(units,activation=None,use_bias=False)
        self.linear_part1 = Dense(units,activation=None,use_bias=True)

    def call(self,x,h,gate_b=None):
        if gate_b is None:
            return tf.nn.sigmoid(self.linear_part1(x)+self.linear_part0(h))
        else:
            return tf.nn.tanh(self.linear_part1(x)+tf.multiply(gate_b,self.linear_part0(h)))

class AUGRU(Layer):
    def __init__(self,units):
        super(AUGRU,self).__init__()
        self.reset_gate = GRU_GATES(units)
        self.update_gate = GRU_GATES(units)
        self.cal_h_part = GRU_GATES(units)

    def call(self,input_x,input_h,attention_score):
        r = self.reset_gate(input_x,input_h)
        u = self.update_gate(input_x,input_h)
        cal_h = self.cal_h_part(input_x,input_h,r)
        u_att = attention_score*u
        h = (1-u_att)*input_h + u_att*cal_h
        return h


class Aux_layer(Layer):
    def __init__(self,hidden_units,activation='relu'):
        super(Aux_layer,self).__init__()
        self.dense_layer=[]
        for unit in hidden_units:
            self.dense_layer.append(Dense(unit,activation))
        self.output_layer = Dense(2,activation=None)

    def call(self,input):
        X=input
        for layer in self.dense_layer:
            X=layer(X)
        output = self.output_layer(X)
        output = tf.nn.softmax(output)
        return output


def build_input_and_embed_layer(feature_columns):
    input_layer_dict = {}
    embed_layer_dict = {}
    for feature in feature_columns:
        if isinstance(feature,SparseFeat):
            input_layer_dict[feature.name]= Input(shape=(1,),name=feature.name+'_input')
            embed_layer_dict[feature.name] = Embedding(feature.vocab_size,feature.embed_dim,mask_zero =True,name=feature.name+'_embed')
        elif isinstance(feature,DenseFeat):
            input_layer_dict[feature.name] = Input(shape=(feature.dimension,),name=feature.name+'_input')
        else:
            input_layer_dict[feature.name] = Input(shape=(feature.behavior_len,),name=feature.name+'_input')
    return input_layer_dict,embed_layer_dict


def cal_aux_loss(aux_layer,input_behavior,input_nonclick,hidden_state,mask):
    input1 = tf.concat([input_behavior,hidden_state],axis=-1)
    input2 = tf.concat([input_nonclick,hidden_state],axis=-1)
    click_prob = aux_layer(input1)[:,:,0]       #[batch_size,max_seq-1,1]
    non_click_prob = aux_layer(input2)[:,:,0]
    click_loss = -tf.reshape(tf.math.log(click_prob),[-1,click_prob.shape[-1]])*mask
    non_click_loss = tf.reshape(tf.math.log(1-non_click_prob),[-1,non_click_prob.shape[-1]])*mask
    aux_loss = tf.reduce_mean(click_loss+non_click_loss)
    return aux_loss






#feature_columns.name:['age','good_id','good_cate','behavior_id','behavior_good',....,'nonclick_id','nonclick_good'],其中good_id\good_cate代指的为候选者的信息
#candidate_profile长度和 behavior_x种类一样多

def DIEN(feature_columns,candidate_profile,behavior_feature_name,neg_behavior_feature_name,attention_hidden_units,attention_activation,
         aux_units,aux_activation,dense_units,dense_activation,alpha):
    input_layer_dict,embed_layer_dict = build_input_and_embed_layer(feature_columns)
    input_list1 = []
    candidate_input = []
    behavior_input_list =[]
    nonclick_input_list = []
    get_mask_vec = []
    get_mask_vec_flag = False
    for feature in feature_columns:
        if isinstance(feature,SparseFeat):
            embed_vec = Flatten()(embed_layer_dict[feature.name](input_layer_dict[feature.name]))
            input_list1.append(embed_vec)
            if feature.name in candidate_profile:
                candidate_input.append(embed_vec)
        elif isinstance(feature,DenseFeat):
            input_list1.append(input_layer_dict[feature.name])
        else:
            if feature.name in behavior_feature_name:
                if not get_mask_vec_flag:
                    get_mask_vec = input_layer_dict[feature.name]
                    get_mask_vec_flag = True
                behavior_input_list.append(embed_layer_dict[feature.sparse_name](input_layer_dict[feature.name]))
            else:
                nonclick_input_list.append(embed_layer_dict[feature.sparse_name](input_layer_dict[feature.name]))

    candidate_input = tf.concat(candidate_input,axis=-1)              #[batch_size,dim]
    input_list2 = tf.concat(input_list1,axis=-1)

    behavior_list = tf.concat(behavior_input_list,axis=-1)             #[batch_size,max_seq,dim]
    nonclick_list = tf.concat(nonclick_input_list,axis=-1)
    gru_dim = behavior_list.shape[-1]
    gru_model = GRU(gru_dim,return_sequences=True)
    behavior_state = gru_model(behavior_list)           #[batch_size,max_seq,dim]
    aux_part = Aux_layer(aux_units,aux_activation)

    mask = tf.cast(tf.not_equal(get_mask_vec,0),dtype=tf.float32)
    aux_loss = cal_aux_loss(aux_part,behavior_list[:,1:,:],nonclick_list[:,1:,:],behavior_state[:,:-1,:],mask[1:])

    attention_layer = Attention_part(attention_hidden_units,attention_activation)
    attention_output = attention_layer(candidate_input,behavior_state,behavior_state,mask)   #[batch_size,dim]

    augru_layer = AUGRU(gru_dim)
    augru_state = tf.zeros_like(behavior_state[:,0,:])

    for idx in range(behavior_state.shape[1]):
        cur_input = tf.squeeze(behavior_state[:,idx,:],axis=1)
        augru_state = augru_layer(cur_input,augru_state,attention_output)

    last_input = tf.concat([input_list2,augru_state],axis=-1)
    deep_layer = DeepPart(dense_units,dense_activation)
    output = deep_layer(last_input)

    model = Model(input_layer_dict,output)
    model.add_loss(alpha*aux_loss)
    return model



if __name__ == '__main__':
    feature_columns4 = [
        SparseFeat('f21', 5, 8),
        SparseFeat('f31', 5, 8),
        SparseFeat('f41', 10, 8),
        SparseFeat('f51', 10, 8),
        behaviorFeat('b1', 'f41', 5),
        behaviorFeat('b2', 'f51', 5),
        behaviorFeat('n1', 'f41', 5),
        behaviorFeat('n2', 'f51', 5)]
    dien = DIEN(feature_columns4, ['f41', 'f51'], ['b1', 'b2'], [], [20, 20], 'prelu', [10, 10], 'relu', [15, 15],
                'prelu', 1.0)
    dien.summary()