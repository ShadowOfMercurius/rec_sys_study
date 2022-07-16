# author:zh

#FM (Factorization Machine,因子分解机)
#引入特征的二阶交叉

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model


class FM_model(Layer):
    def __init__(self,v_dim,reg1=0.01,reg2=0.01):
        super(FM_model,self).__init__()
        self.v_dim = v_dim
        self.reg1 = reg1
        self.reg2 = reg2

    def build(self,input_shape):
        self.w0 = self.add_weight(name='w0',shape=(1,),initializer=tf.zeros_initializer(),trainable=True)
        self.w = self.add_weight(name='w',shape=(input_shape[-1],1),initializer=tf.random_normal_initializer(),
                                 trainable=True,regularizer=tf.keras.regularizers.l2(self.reg1))
        self.v = self.add_weight(name='v',shape=(input_shape[-1],self.v_dim),initializer=tf.random_normal_initializer(),
                                 trainable=True,regularizer=tf.keras.regularizers.l2(self.reg2))

    def call(self,input):
        inter_part1 = tf.pow((tf.matmul(input,self.v)),2)
        inter_part2 = tf.matmul(tf.pow(input,2),tf.pow(self.v,2))
        linear_part = self.w0+tf.matmul(input,self.w)
        inter_part = tf.reduce_sum(inter_part1-inter_part2,axis=-1,keepdims=True)
        output = linear_part+inter_part
        return output





