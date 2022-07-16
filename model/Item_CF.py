# author:zh


from collections import defaultdict
import math
def ItemCF_get_similarity(input):
    '''
    :param input: Dict[List]，物品：{user1,user2,...}
    :return: sim_martrix:相似度矩阵
    '''
    #step1 得到倒排表
    matrix = defaultdict(set)
    for item,users in input.items():
        for user in users:
            matrix[user].add(item)
    #step2 得到物品共现表,即物品item1，item2被同时喜欢的次数
    matrix2 = defaultdict(int)
    item_cnts_matrix = defaultdict(int)
    for user,items in matrix.items():
        for item1 in items:
            item_cnts_matrix[item1] += 1
            for item2 in items:
                if item1 == item2:
                    continue
                else:
                    matrix2[(item1,item2)] += 1
    #step3 得到相似度计算
    sim_matrix = defaultdict(dict)
    for co_item,cnts in matrix2.items():
        item1,item2 = co_item[0],co_item[1]
        sim_matrix[item1][item2] = cnts/math.sqrt(item_cnts_matrix[co_item[0]]*item_cnts_matrix[co_item[1]])
    return sim_matrix


def ItemCf_recomend(target,sim_matrix,sim_item_K,target_user_item_list,recommend_top_k):
    '''
    :param target: 目标用户
    :param sim_matrix: 物品相似矩阵
    :param sim_item_K: 物品相似度的topk
    :param target_user_item_list: 目标用户的已交互item
    :param recommend_top_k: 推荐列表的topk
    :return:recommend_list:最终推荐结果，List([score,item])
    '''

    item_list = target_user_item_list[target]
    rank = defaultdict(int)
    related_rank = dict()
    #找出item列表中每个有过交互的item的topK
    for item in item_list:
        item_related = sim_matrix[item]   #item和其他的相关度
        item_related_list = [(score,item2) for item2,score in item_related.items()]
        item_related_list = sorted(item_related_list, reverse=True)
        item_related_list2 = []
        start = 0
        cnt = sim_item_K
        while start < len(item_related_list) and cnt > 0:
            if item_related_list[start] in item_list:
                start += 1
                continue
            else:
                item_related_list2.append(item_related_list[start])
                start += 1
                cnt -= 1
        related_rank[item] = item_related_list2  #{item:[(score,item1).....最相关的k个]}
    top_k_item=defaultdict()
    for item,related_info in related_rank.items():
        for score,item_name in related_info:
            top_k_item[item_name] += score     #如果item的权重不同，在此处修改score*rate

    recommend_list = sorted([(score,item_name) for score,item_name in top_k_item.items()],reverse=True)[0:recommend_top_k]
    return recommend_list

