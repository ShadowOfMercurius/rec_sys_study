# author:zh

#DIN(Deep Interst Network)：深度兴趣网络
#对比传统embedding+mlp模型，在对用户历史信息处理采用attention机制，从而实现了当前目前找到相关的历史行为（代表兴趣）而非传统的统一处理

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten,BatchNormalization,PReLU
from tensorflow.keras import Model
from collections import namedtuple
from wide_deep import wide_part,deep_part
from utils import SparseFeat,DenseFeat,behaviorFeat


class Dice(Layer):
    def __init__(self):
        super(Dice,self).__init__()
        self.bn_layer = BatchNormallization(center=False,scale=False)
        self.alpha = self.add_weight(name='alpha',shape=(1,),trainable=True)

    def call(self,input):
        x = self.bn_layer(input)
        x = tf.nn.sigmoid(x)
        output = x*(input) + (1-x)*self.alpha*input
        return output


class Attention_part(Layer):
    def __init__(self,hidden_units,activation='prelu'):
        super(Attention_part,self).__init__()
        self.dense_layer=[]
        for unit in hidden_units:
            if activation=='prelu':
                self.dense_layer.append(Dense(unit,activation=PReLU()))
            else:
                self.dense_layer.append(Dense(unit,activation=Dice()))
        self.output_layer = Dense(1,activation=None)

    def call(self,query,key,value,mask=None):
        query = tf.expand_dims(query,axis=1)     #[batch_size,embed_dim]->[batch_size,1,embed_dim]
        query = tf.tile(query,[1,key.shape[1],1]) #[batch_size,keys_len,embed_dim]
        embed = tf.concat([query,key,query-key,query*key],axis=-1)
        x = embed
        for layer in self.dense_layer:
            x = layer(x)
        score = self.output_layer(x)       #[batch_size,keys_len,1]
        score = tf.squeeze(score,axis=-1)
        if mask:
            padding = tf.ones_like(score)
            score = tf.where(tf.equal(mask,0),padding,score)
        score2 = tf.expand_dims(score,axis=1)
        output = tf.matmul(score2,value)
        output =tf.squeeze(output,axis=1)   #[batch_size,embed_dim]
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



class DeepPart(Layer):
    def __init__(self,hidden_units,activation='prelu'):
        super(DeepPart,self).__init__()
        self.dense_layer=[]
        for unit in hidden_units:
            if activation=='prelu':
                self.dense_layer.append(Dense(unit,activation=PReLU()))
            else:
                self.dense_layer.append(Dense(unit,activation=Dice()))
        self.output_layer = Dense(1,activation='sigmoid')

    def call(self,input):
        x = input
        for layer in self.dense_layer:
            x = layer(x)
        return self.output_layer(x)



#feature_columns.name:['age','good_id','good_cate','behavior_id','behavior_good',....],其中good_id\good_cate代指的为候选者的信息
#candidate_profile长度和 behavior_x种类一样多
def DIN(feature_columns,candidate_profile,attention_hidden_units,attention_activation,dense_units,dense_activation):
    input_layer_dict,embed_layer = build_input_and_embed_layer(feature_columns)
    input_list1 = []
    behavior_list =[]
    candidate_input = []
    get_mask_vec_flag= False
    for feature in feature_columns:
        if isinstance(feature,SparseFeat):
            embed_vec = Flatten()(embed_layer[feature.name](input_layer_dict[feature.name]))
            input_list1.append(embed_vec)
            if feature.name in candidate_profile:
                candidate_input.append(embed_vec)
        elif isinstance(feature,DenseFeat):
            input_list1.append(input_layer_dict[feature.name])
        else:
            if not get_mask_vec_flag:
                get_mask_vec = input_layer_dict[feature.name]
                get_mask_vec_flag= True
            behavior_list.append(embed_layer[feature.sparse_name](input_layer_dict[feature.name]))

    candidate_input = tf.concat(candidate_input,axis=-1)   #[batch_size,embed_dim*num_candidate_profile]
    input_list2= tf.concat(input_list1,axis=-1)  #非行为部分
    behavior_list2 = tf.concat(behavior_list,axis=-1)   #[batch_size,max_seq,embed_dim*num_candidate_profile]
    help_mask_vec = get_mask_vec
    mask = tf.cast(tf.not_equal(help_mask_vec,0),dtype=tf.float32)    #batch_size,max_seq
    attention_layer = Attention_part(attention_hidden_units,attention_activation)
    attention_output = attention_layer(candidate_input,behavior_list2,behavior_list2,mask)
    last_input = tf.concat([attention_output,input_list2],axis=-1)
    mlp_layer = DeepPart(dense_units,dense_activation)
    output = mlp_layer(last_input)
    model = Model(input_layer_dict,output)
    return model





if __name__=='__main__':
    feature_columns3 = [
        SparseFeat('f21', 5, 8),
        SparseFeat('f31', 5, 8),
        SparseFeat('f41', 10, 8),
        SparseFeat('f51', 10, 8),
        behaviorFeat('b1', 'f41', 5),
        behaviorFeat('b2', 'f51', 5)]
    din = DIN(feature_columns3, ['f41', 'f51'], [20, 20], 'prelu', [10, 10], 'prelu')
    din.summary()
    tf.keras.utils.plot_model(din, to_file='din.png', show_shapes=True)