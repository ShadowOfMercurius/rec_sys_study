# author:zh

#wide&deep 由wide浅层结构（memorization 记忆）和deep深层结构（generalization 泛化）构成

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten
from tensorflow.keras import Model
from collections import namedtuple



SparseFeat = namedtuple('SparseFeat',['name','vocab_size','embed_dim'])



class wide_part(Layer):
    def __init__(self):
        super(wide_part,self).__init__()

    def build(self,input_shape):
        self.w0 = self.add_weight(name='w0',shape=(1,),initializer=tf.zeros_initializer(),trainable=True)
        self.w = self.add_weight(name='w',shape=(input_shape[-1],1),initializer=tf.random_normal_initializer(),
                                 trainable=True,regularizer=tf.keras.regularizers.l1(1e-4))

    def call(self,input):
        x = tf.matmul(input,self.w)+self.w0
        return x

class deep_part(Layer):
    def __init__(self,hidden_units,output_dim,activation='relu'):
        super(deep_part,self).__init__()
        self.dense_layer = []
        for unit in hidden_units:
            self.dense_layer.append(Dense(unit,activation,kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        self.output_layer = Dense(output_dim,activation=None)

    def call(self,input):
        x = input
        for layer in self.dense_layer:
            x = layer(x)
        return self.output_layer(x)


def Wide_Deep(wide_input_length,deep_feature,deep_units,deep_dim=1):
    input_layer_dict={}
    embed_layer_dict ={}
    deep_input = []
    for feature in deep_feature:
        if isinstance(feature,SparseFeat):
            input_layer_dict[feature.name] = Input(shape=(1,),name=feature.name)
            embed_layer_dict[feature.name] = Flatten()(Embedding(feature.vocab_size,feature.embed_dim)(input_layer_dict[feature.name]))
            deep_input.append(embed_layer_dict[feature.name])
        else:
            input_layer_dict[feature] = Input(shape=(1,),name=feature)
            deep_input.append(input_layer_dict[feature])
    input_layer_dict['wide_input'] = Input(shape=(wide_input_length,1))
    deep_input = tf.concat(deep_input,axis=-1)
    wide_layer = wide_part()
    deep_layer = deep_part(deep_units,deep_dim)
    wide_output = wide_layer(input_layer_dict['wide_input'])
    deep_output = deep_layer(deep_input)
    res = tf.nn.sigmoid(0.5*(wide_output+deep_output))
    model = Model(input_layer_dict,res)
    return model


if __name__ == '__main__':
    input = {'wide_input': np.array([[1, 3], [4, 5], [2, 3], [3, 3], [2, 1]]),
             'f11': np.array([5, 2, 1, 4, 9]),
             'f21': np.array([2, 1, 4, 2, 2]),
             'f31': np.array([4, 2, 1, 3, 3])}
    label = np.array([1, 0, 0, 0, 1])
    feature_columns = ['f11',
                       SparseFeat('f21', 5, 8),
                       SparseFeat('f31', 5, 8),
                       ]
    model = Wide_Deep(2,feature_columns,[20,20])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    model.fit(input, label, epochs=5, batch_size=32)


