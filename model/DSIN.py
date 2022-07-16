# author:zh

#DSIN：Deep session interest network
#相比其他的针对行为建模，DSIN对于session进行建模，认为session内的行为高度同构、而不同session内的行为不同

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten,Bidirectional,LSTM,LayerNormalization,Dropout
from tensorflow.keras import Model
from collections import namedtuple
from utils import SparseFeat,DenseFeat,behaviorFeat
from DIN import Attention_part,Dice,DeepPart


def attention_cal(query,key,value,mask):
    matmul_qk = tf.matmul(query,key,transpose_b=True)
    d_model = key.shape[-1]
    matmul_qk = matmul_qk/tf.math.sqrt(tf.cast(d_model,dtype=tf.float32))
    if mask is not None:
        matmul_qk += (mask*1e-9)
    attention_score = tf.nn.softmax(matmul_qk,axis=-1)
    output = tf.matmul(attention_score,value)
    return output,attention_score


class MultiHead_Attention(Layer):
    def __init__(self,d_model,num_heads):
        super(MultiHead_Attention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model %num_heads ==0
        self.each_head_size = d_model//num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense_layer = Dense(d_model)

    def split(self,inputs,batch_size):
        x = tf.reshape(inputs,(batch_size,-1,self.num_heads,self.each_head_size))
        x = tf.transpose(x,[0,2,1,3])
        return x

    def call(self,query,key,value,mask):
        batch_size = query.shape[0]
        query_matrix = self.wq(query)
        key_matrix = self.wk(key)
        value_matrix = self.wv(value)

        q = self.split_heads(query_matrix, batch_size)
        k = self.split_heads(key_matrix, batch_size)
        v = self.split_heads(value_matrix, batch_size)

        att_output, attention_score = attention_cal(q, k, v, mask)              #att_output = [batch_size,num_heads,max_sess,input_shape[-1]
        att2 = tf.transpose(att_output, [0, 2, 1, 3])
        mlp_input = tf.reshape(att2, (batch_size, -1, self.d_model))            #多头合并   [batch_size,max_sess,d_models]
        output = self.dense(mlp_input)
        return output


class feed_forward_network(Layer):
    def __init__(self,d_model,hidden_list):
        super(feed_forward_network).__init__()
        self.hidden_list = hidden_list
        self.d_model = d_model
        self.hidden_layer = [Dense(node_cnt,activation='relu') for node_cnt in self.hidden_list]
        self.output_layer = Dense(d_model,activation=None)

    def call(self,inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output


class Interest_Extractor_Part(Layer):
    def __init__(self,d_model,num_heads,hidden_units,dropout_rate=0.1):
        self.self_attention_part = MultiHead_Attention(d_model,num_heads)
        self.ffn = feed_forward_network(d_model,hidden_units)
        self.Ln1 = LayerNormalization()
        self.Ln1 = LayerNormalization()
        self.Ln2 = LayerNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self,inputs,mask):
        att_output,score = self.self_attention_part(inputs,inputs,inputs,mask)
        ffn_input = self.dropout1(att_output)
        ffn_input = self.Ln1(ffn_input+inputs)  #类残差结构
        ffn_output = self.ffn(ffn_input)   #MLP
        ffn_output = self.dropout2(ffn_output)
        out = self.Ln2(ffn_output+ffn_input)   #类残差结构
        out2 = tf.reduce_mean(out,axis=1,keepdims=True) #[batch_size,1,d_model]
        return out2


# feature_columns.name:['age','good_id','good_cate','session1_id','session1_good_id','.....']

sessionFeat = namedtuple('sessionFeat', ['name', 'sparse_name', 'session_no', 'session_len']) #('name','good_id','sess_1','10')

def build_input_and_embed_layer(feature_columns):
    input_layer_dict = {}
    embed_layer_dict = {}
    for feature in feature_columns:
        if isinstance(feature, SparseFeat):
            input_layer_dict[feature.name] = Input(shape=(1,), name=feature.name + '_input')
            embed_layer_dict[feature.name] = Embedding(feature.vocab_size, feature.embed_dim, mask_zero=True,
                                                       name=feature.name + '_embed')
        elif isinstance(feature, DenseFeat):
            input_layer_dict[feature.name] = Input(shape=(feature.dimension,), name=feature.name + '_input')
        else:
            input_layer_dict[feature.name] = Input(shape=(feature.session_len,), name=feature.name + '_input')
    return input_layer_dict, embed_layer_dict

def concat_sess_input(sess_input_dic,mask_input,sess_cnt,feature_order):
    res = []
    mask_list = []
    for i in range(sess_cnt):
        key ='sess_'+str(i+1)
        cur_list = []
        for feature in feature_order:
            cur_list.append(sess_input_dic[key][feature])
        mask_list.append(mask_input[key])
        res.append(tf.concat(cur_list,axis=-1))
    return res,mask_list                   #res:LIST,共sess_cnt个元素，每个元素为[batch_size,max_sess_len,embed_dim*len(feature_order)].mask_list:sess_cnt个元素，[batch_size,max_sess_len]


def create_mask_session(mask_sessions,sess_cnt):
    res = []
    for i in range(sess_cnt):
        key = 'sess_' + str(i + 1)
        res.append(mask_sessions[key])
    res = tf.concat(res,axis=0)
    res = tf.transpose(res,[1,0])
    mask = tf.cast(tf.not_equal(res,0),tf.float32)
    return mask


def DSIN(feature_columns,candidate_profile,session_cnt,num_heads,hidden_units,dropout_rate,attention_units,attention_activation,dense_units,dense_activation):
    input_layer_dict, embed_layer_dict = build_input_and_embed_layer(feature_columns)
    input_list1 = []
    candidate_input = []
    sess_input = {}
    mask_input = {}
    mask_sessions = {}
    for feature in feature_columns:
        if isinstance(feature,SparseFeat):
            embed_vec = Flatten()(embed_layer_dict[feature.name](input_layer_dict[feature.name]))
            input_list1.append(embed_vec)
            if feature.name in candidate_profile:
                candidate_input.append(embed_vec)
        elif isinstance(feature,DenseFeat):
            input_list1.append(input_layer_dict[feature.name])
        else:
            session_no = feature.session_no
            if session_no not in mask_input.keys():
                tmp = input_layer_dict[feature.name]
                mask = tf.cast(tf.not_equal(tmp,0),dtype=tf.float32)
                mask_input[session_no]=mask
                mask_sessions[session_no] = tmp[:,0]

            embed_vec = embed_layer_dict[feature.sparse_name](input_layer_dict[feature.name])   #[batch_size,max_each_sess_len,embed_dim]
            if session_no not in sess_input.keys():
                sess_input[session_no]={}
                sess_input[session_no][feature.sparse_name] = embed_vec
            else:
                sess_input[session_no][feature.sparse_name] = embed_vec

    non_sess_input = tf.concat(input_list1,axis=-1)
    candidate_input2 = tf.concat(candidate_input,axis=-1)

    sess_input2,mask_list = concat_sess_input(sess_input,mask_input,session_cnt,candidate_profile)            #list,共session_CNT数目个元素，每个元素是[batch_size,max_each_sess_len,embed_dim]

    d_model = candidate_input2.shape[0]

    interest_extractor = Interest_Extractor_Part(d_model,num_heads,hidden_units,dropout_rate)

    sess_interest_extractor_list = []
    for i in range(session_cnt):
        att_scr = interest_extractor(sess_input2[i],mask_list[i])               #[batch_size,1,d_model]
        sess_interest_extractor_list.append(att_scr)
    sess_interest = tf.concat(sess_interest_extractor_list,axis=1)             #[batch_size,sess_num,d_model]

    bilstm_part = Bidirectional(LSTM(d_model,return_sequences=True),merge_model='ave')
    sess_interest_interacting = bilstm_part(sess_interest)               #[batch_size,sess_num_d_model]

    attention_part1 = Attention_part(attention_units,attention_activation)
    attention_part2 = Attention_part(attention_units,attention_activation)
    mask_sess = create_mask_session(mask_sessions, session_cnt)
    interest_vec1 = attention_part1(candidate_input2,sess_interest,sess_interest,mask_sess)
    interest_vec2 = attention_part2(candidate_input2,sess_interest_interacting,sess_interest_interacting,mask_sess)

    last_input = tf.concat([non_sess_input,interest_vec1,interest_vec2],axis=-1)
    deep_layer = DeepPart(dense_units,dense_activation)
    output = deep_layer(last_input)
    model = Model(input_layer_dict,output)
    return model


