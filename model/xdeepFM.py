# author:zh

#xDeepFM 并非是deepFM的演变，更像是DCN模型的进一步改进，通过对DCN中的cross部分进行改进，引入域的概念（压缩感知）


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten,Conv1D
from tensorflow.keras import Model
from collections import namedtuple
from wide_deep import deep_part,wide_part
from utils import build_input_dict_for_widedeep,SparseFeat


class CIN(Layer):
    def __init__(self,cin_size_list):
        super(CIN,self).__init__()
        self.cin_size_list=cin_size_list

    def build(self,input_shape):
        #input_shape:[None,n,k]  n:特征域个数，k维度
        self.field_num = [input_shape[1]]+self.cin_size_list
        self.cin_W = []
        for i in range(len(self.field_num)-1):
            w = self.add_weight(name='w'+str(i),shape=(1,self.field_num[i]*self.field_num[0],self.field_num[i+1]),
                            initializer=tf.keras.initializers.glorot_uniform(),
                            regularizer=tf.keras.regularizers.l1_l2(1e-5),
                            trainable=True)
            self.cin_W.append(w)

    def call(self,inputs):
        dim = inputs.shape[-1]
        x0 = tf.split(inputs,dim,axis=-1)       #切分成一个list，长度为dim，里面每个元素为tensor[batch_size,feature_dim,1]
        each_layer_res=[inputs]
        for idx,size in enumerate(self.field_num[1:]):    #field_num 共n+1层 0~第一层大小/1～第2层大小   n-1/第n层大小  （不包含第0层大小）
            pre_output = each_layer_res[-1]
            xi = tf.split(pre_output,dim,axis=-1)
            x = tf.matmul(x0,xi,transpose_b=True)   #list长度不变为dim，里面每个元素[batch_size,x0的feature_dim,idx的feature_dim]
            x = tf.reshape(x,shape=[dim,-1,self.field_num[0]*self.field_num[idx]])   #[dim,batch_size,f_dim0*f_dim_idx]
            x = tf.transpose(x,[1,0,2])              #[batch_size,dim,f_dim0*f_dim_x1]
            x = tf.nn.conv1d(input=x,filters=self.cin_W[idx],stride=1,padding='VALID')
            x = tf.transpose(x,[0,2,1])
            each_layer_res.append(x)
        each_layer_res = each_layer_res[1:]
        output = tf.concat(each_layer_res,axis=1)
        output = tf.reduce_sum(output,axis=-1)
        return output



def xDeepFM(linear_input_dim,feature_columns,hidden_units,deep_output_dim,cin_size_list,activation='relu'):
    input_layer_dict = {}
    input_layer_dict['lr_input'] = Input(shape=(linear_input_dim,))
    embed_layer_dict={}
    embed_list=[]
    deep_input_list=[]
    lr_input_list = [input_layer_dict['lr_input']]
    for feature in feature_columns:
        if isinstance(feature,SparseFeat):
            input_layer_dict[feature.name] = Input(shape=(1,))
            embed_layer_dict[feature.name] = Flatten()(Embedding(feature.vocab_size,feature.embed_dim)(input_layer_dict[feature.name]))
            embed_list.append(embed_layer_dict[feature.name])      #list,feature_nums*[batch_size,embed_dim]
            deep_input_list.append(embed_layer_dict[feature.name])
        else:
            input_layer_dict[feature] = Input(shape=(1,))
            deep_input_list.append(input_layer_dict[feature])
            lr_input_list.append(input_layer_dict[feature])
    lr_input = tf.concat(lr_input_list,axis=-1)
    deep_input = tf.concat(deep_input_list,axis=-1)
    cin_input = tf.transpose(tf.convert_to_tensor(embed_list),[1,0,2])
    lr_layer = wide_part()
    deep_layer = deep_part(hidden_units,deep_output_dim,activation)
    cin_layer = CIN(cin_size_list)
    lr_output = lr_layer(lr_input)
    deep_output = deep_layer(deep_input)
    cin_output = cin_layer(cin_input)
    output = tf.concat([lr_output,deep_output,cin_output],axis=-1)
    #output = lr_output+deep_output+cin_output
    output2 = Dense(1,activation='sigmoid')(output)
    model = Model(input_layer_dict,output2)
    return model



if __name__ == '__main__':
    feature_columns = ['f1',
                       SparseFeat('f2', 'user_id', 5, 8),
                       SparseFeat('f3', 'user_id', 5, 8),
                       ]
    xdf = xDeepFM(4, feature_columns, [20, 20], 1, [5, 10])
    xdf.summary()







