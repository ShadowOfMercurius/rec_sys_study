# author:zh

#LFM（Latent Factor Model,隐语义模型,利用MF技术（Matrix Factorization）

import random
import math


class Latent():
    def __init__(self,data,latent_size=6,alpha=0.1,beta=0.1,max_iter = 50):
        self.latent_size=latent_size
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.user_latent_matrix=dict()
        self.item_latent_matrix=dict()
        self.user_bias = dict()
        self.item_bias = dict()
        self.global_bias = 0.0   #非训练因子
        cnt = 0
        for user,items in self.data.items():
            #经验上随机数和1/sqrt(latent_size)成正比的随机化最好
            self.user_latent_matrix[user] = [random.random()/math.sqrt(self.latent_size) for _ in range(self.latent_size)]
            self.user_bias[user] = 0
            cnt = len(items)
            for item,rating in items.items():
                self.global_bias += rating
                if item not in self.item_latent_matrix:
                    self.item_latent_matrix[item] = [random.random()/math.sqrt(self.latent_size) for _ in range(self.latent_size)]
                    self.item_bias[item] = 0
        self.global_bias = self.global_bias/cnt

    def train(self):
        for step in range(self.max_iter):
            for user,items in self.data.items():
                for item,rating in items:
                    predict_rating = self.predict(user,item)
                    error = rating-predict_rating
                    self.user_bias[user] += self.alpha *(error - self.beta*self.user_bias[user])
                    self.item_bias[item] += self.alpha*(error-self.beta*self.item_bias[item])
                    for k in range(0,self.latent_size):
                        self.user_latent_matrix[user][k] += self.alpha *(error*self.item_latent_matrix[item][k] - self.beta*self.user_latent_matrix[user][k])
                        self.item_latent_matrix[item][k] += self.alpha *(error*self.user_latent_matrix[user][k] - self.beta*self.item_latent_matrix[item][k])
            self.alpha *= 0.05


    def predict(self,user,item):
        score = 0
        for i in range(0,self.latent_size):
            score += self.user_latent_matrix[user][i]*self.item_latent_matrix[item][i]
            score += self.user_bias[user]+self.item_bias[item]+self.global_bias
        return score











