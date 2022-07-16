# author:zh

#deep_crossing,将sparse feature进行embedding后和dense feature拼接送入MLP

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten
from tensorflow.keras import Model
from collections import namedtuple

SparseFeat = namedtuple('SparseFeat',['name','vocab_size','embed_dim'])

def deep_crossing_model(feature_columns,hidden_units,activation='relu'):
    input_layer_dict={}
    embed_columns = []
    dense_columns=[]
    for feature in feature_columns:
        if isinstance(feature,SparseFeat):
            input_layer_dict[feature.name] = Input(shape=(1,),name=feature.name)
            embed_columns.append(feature)
        else:
            input_layer_dict[feature] = Input(shape=(1,),name=feature,dtype='float32')
            dense_columns.append(feature)
    embed_layer_dict = {}
    mlp_input_list=[]
    print(embed_columns)
    for name in dense_columns:
        mlp_input_list.append(input_layer_dict[name])
    for feature in embed_columns:
        embed_layer_dict[feature.name] = Flatten()(Embedding(feature.vocab_size,feature.embed_dim)(input_layer_dict[feature.name]))
        mlp_input_list.append(embed_layer_dict[feature.name])

    mlp_input = tf.concat(mlp_input_list,axis=-1)
    x=mlp_input
    for idx,unit in enumerate(hidden_units):
        x = Dense(unit,activation,name='dense_'+str(idx))(x)
    output = Dense(1,activation='sigmoid',name='output_layer')(x)
    model = Model(input_layer_dict,output)
    return model

if __name__ == '__main__':
    data = pd.DataFrame()
    #input label仅供测试
    input = {'f1':np.array([5,2,1,4,9]),
             'f2':np.array([2,1,4,2,2]),
             'f3':np.array([4,2,1,3,3])}
    label = np.array([1,0,0,0,1])
    feature_embed = ['f1',
                     SparseFeat('f2', 'user_id', 5, 8),
                     SparseFeat('f3', 'user_id', 5, 8),
                     ]
    '''
    feature_embed = ['f1',
                     SparseFeat('f2', 'user_id', max(data['user_id']) + 1, 8),
                     SparseFeat('f3', 'user_id', max(data['user_id']) + 1, 8),
                     ]
    '''

    model = deep_crossing_model(['f1','f2','f3'],[10,10])
    model.summary()
    tf.keras.utils.plot_model(model, to_file='neumf.png', show_shapes=True)
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    model.fit(input, label, epochs=5,batch_size=32)
    model.predict(input)




