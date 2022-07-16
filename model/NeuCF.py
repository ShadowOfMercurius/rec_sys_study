# author:zh

#NeuCF (Neural Collaborative Filtering),分别对user向量和item向量使用embedding技术后利用技术进行输出
#传统NCF在融合上采用的方法包括element-wise product、concatenation两类操作
#NeuMF（Neural Matrix Factorization）则是将上述两种方法合并，此处代码直接使用NeuMF，注意NeuMF对于user生成了对于GMF和MLP的两个向量
#对item也是生成GMF和MLP两个向量

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten
from tensorflow.keras import Model
from collections import namedtuple
import pandas as pd

SparseFeat = namedtuple('SparseFeat',['name','input_columns','vocab_size','embed_dim'])

def NeuralMF_Model(feature_columns,feature_embed,hidden_units,activation='relu'):
    input_layer_dict={}
    embed_layer_dict={}
    for feature in feature_columns:
        input_layer_dict[feature] = Input(shape=(1,),name=feature+'input_layer')

    for feature in feature_embed:
        embed_layer_dict[feature.name] = \
            Flatten()(Embedding(feature.vocab_size,feature.embed_dim,name=feature.name)(input_layer_dict[feature.input_columns])) #embed后[batch,feature_dim]->[batch_feature_dim,embed_dim],此时dim=1需要铺平

    gmf_input = tf.multiply(embed_layer_dict['user_id_mf'],embed_layer_dict['item_id_mf'])
    mlp_input = tf.concat([embed_layer_dict['user_id_mlp'],embed_layer_dict['item_id_mlp']],axis=-1)

    x1 = mlp_input
    for idx,unit in enumerate(hidden_units):
        x1 = Dense(unit,activation=activation,name='dense_'+str(idx))(x1)
    output = Dense(1,activation='sigmoid',name='ouput_layer')(tf.concat([gmf_input,x1],axis=-1))
    model = Model(input_layer_dict,output)
    return model




if __name__ == '__main__':
    data = pd.DataFrame()
    #input label仅供测试
    input = {'user_id':np.array([5,5,5,10,10,10]),'item_id':np.array([3,4,5,3,4,5])}
    label = np.array([1,0,0,0,0,1])
    feature_embed = [SparseFeat('user_id_mf', 'user_id', max(data['user_id']) + 1, 8),
                     SparseFeat('user_id_mlp', 'user_id', max(data['user_id']) + 1, 8),
                     SparseFeat('item_id_mf', 'item_id', max(data['item_id']) + 1, 8),
                     SparseFeat('item_id_mlp', 'item_id', max(data['item_id']) + 1, 8),
                     ]
    model = NeuralMF_Model(['user_id','item_id'],feature_embed,[10,10])
    model.summary()
    tf.keras.utils.plot_model(model, to_file='neumf.png', show_shapes=True)
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    model.fit(input, label, epochs=5,batch_size=32)
    model.predict(input)









