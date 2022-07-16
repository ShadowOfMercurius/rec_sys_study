from collections import namedtuple,OrderedDict
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Input,Embedding,Flatten

SparseFeat = namedtuple('SparseFeat',['name','vocab_size','embed_dim',])
DenseFeat = namedtuple('DenseFeat',['name','dimension'])
behaviorFeat = namedtuple('behaviorFeat',['name','sparse_name','behavior_len'])  #包括
sessionFeat = namedtuple('sessionFeat',['name','sparse_name','session_no','session_len'])




#此函数只是简单示例，例如非deep部分也可以由embedding层后的输入组合,仿照deep部分修改输入层代码即可
def build_input_dict_for_widedeep(not_deep_input_dim,deep_input_feature,not_deep_input_name):
    input_layer_dict={}
    embed_layer_dict={}
    deep_input_list =[]
    input_layer_dict[not_deep_input_name] = Input(shape=(not_deep_input_dim,))
    for feature in deep_input_feature:
        if isinstance(feature,SparseFeat):
            input_layer_dict[feature.name] = Input(shape=(1,))
            embed_layer_dict[feature.name] = Flatten()(
                Embedding(feature.vocab_size, feature.embed_dim)(input_layer_dict[feature.name]))
            deep_input_list.append(embed_layer_dict[feature.name])
        else:
            input_layer_dict[feature] = Input(shape=(1,))
            deep_input_list.append(input_layer_dict[feature])
    deep_input = tf.concat(deep_input_list,axis=-1)
    not_deep_input = input_layer_dict[not_deep_input_name]
    return not_deep_input,deep_input,input_layer_dict




