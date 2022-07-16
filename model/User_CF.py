# author:zh


from collections import defaultdict
import math

def UserCF_get_similarity(input):
    '''
    :param input:  Dict[List]，记录行为数据，如User_CF记录，每个元素记录的是当前用户和起对应的物品dict['A']=['a','b','c']
    :return:sim_matrix：相似度矩阵
    '''
    #step1.得到用户的倒查表
    matrix = defaultdict(set)  #matrix为倒查表
    for user,items in input.items():
        for item in items:
            matrix[item].add(user)
    #step2.得到用户的共现矩阵等信息，即用户i和用户j同时喜欢某个物品的次数
    matrix2 = defaultdict(int)       #matrix2为记录用户之间的共现矩阵
    behavior_nums_matrix = defaultdict(int)
    for item,users in matrix.items():
        for user_no1 in users:
            behavior_nums_matrix[user_no1] += 1   #计算用户有行为的数目，之后计算相似度分母需要
            for user_no2 in users:
                if user_no1 == user_no2:
                    continue
                else:
                    matrix2[(user_no1,user_no2)]+=1
    #step3 计算相似度，此处为最简单的计算公式，有关改进也可在这步进行修正
    sim_matrix = defaultdict(int)
    for co_user,cnt in matrix2.items():
        sim_matrix[co_user] = cnt/math.sqrt(behavior_nums_matrix[co_user[0]] * behavior_nums_matrix[co_user[1]])
    return sim_matrix

def User_CF_recommend(target,sim_matrix,sim_user_K,input,recommend_top_k):
    '''
    :param target: 待推荐用户
    :param sim_matrix: 相似度矩阵
    :param sim_user_K: 相似度排名前K的用户
    :param input: 记录了用户input-item的关联信息
    :param recommend_top_k: 输出推荐最相关的topK
    :return:res 包含了top K个相关的推荐，List[(相关度，物品)]，从大至小排列
    '''
    related_user = []
    rank = defaultdict(int)
    #找出相关的所有用户
    for (user1,user2),score in sim_matrix.items():
        if user1 == target:
            if score !=0:
                related_user.append([score,user2])
    related_user = sorted(related_user,reverse=True)[0:sim_user_K]   #根据分数从大到小排序
    #根据相关的K个用户找出最相关的物品
    for score,user in related_user:
        related_user_behavior = input[user]
        for item in related_user_behavior:
            if item in input[target]:   #若是已交互过的item不再推荐则保留，否则不需要此处if
                continue
            rank[item] = score #此处将用户对所有物品的兴趣度视作1，若是不同则为score*rate，rate为兴趣度
    recommend_list_all = []
    for item,score in rank.items():
        recommend_list_all.append([score,item])
    res = sorted(recommend_list_all,reverse=True)[0:recommend_top_k]
    return res








