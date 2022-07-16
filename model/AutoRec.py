# author:zh

#AutoRec:
# 自编码器和CF的结合，利用CF中的共现矩阵，完成item or User的自编码，再利用自编码得到用户对物品的评分预估，输入为评分向量，空值考虑默认值或者平均值填充

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input
from tensorflow.keras import Model



#利用function性质进行定义
def AutoRec_Model(input_shape,output_shape,hidden_units,activation='relu'):
    input_layer = Input(shape=(input_shape,),name='input_layer')
    x = input_layer
    for idx,unit in enumerate(hidden_units):
        x = Dense(unit,activation='relu',name='dense_layer_'+str(idx))(x)
    output_layer = Dense(output_shape,activation=None,name='output_layer')(x)
    model = Model(input_layer,output_layer)
    return model


if __name__ =='__main__':
    input = np.array([])
    hidden_units = []
    input_shape = input.shape[-1]
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model = AutoRec_Model(input_shape,input_shape,hidden_units)
    #summary型总结
    model.summary()
    #画出结构图
    tf.keras.utils.plot_model(model,to_file='',show_shapes=True)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    model.fit(input, input, epochs=100, validation_split=0.1, verbose=0)
    #评估(根绝complie中的输入，输出loss，metrics，在此会输出mse，mae，mse)
    evaluate_output = model.evaluate(input,input)
    #预测
    predict_value = model.predict(input)




'''
利用自定义类进行实现
class AutoRec(Model):
    def __init__(self,hidden_units,activation='relu'):
        super(AutoRec,self).__init__()
        self.hidden_units=hidden_units
        self.dense_layer = []
        for unit in hidden_units:
            self.dense_layer.append(Dense(unit,activation))

    def call(self,input):
        x=input
        for layer in self.dense_layer:
            x = layer(x)
        return x

if __name__=='__main__':
    input = np.array([])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model = AutoRec(hidden_units=[])

    model.complie(loss='mse',optimizer=optimizer,metrics=['mae','mse'])

    model.fit(input,input,epochs=100,validation_split = 0.1,verbose=0)

    predict_value = model.predict(input)
'''