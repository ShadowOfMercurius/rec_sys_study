# rec_sys_study
model部分为学习推荐系统中的一些结构的简单实现,包括以下部分

1.协同过滤（Collaborative Filtering,CF)：利用用户和物品的交互信息,对相似的用户(User_CF)或是相同的物品(Item_CF)进行推荐

2.隐语义模型(Latent Factor Model,LFM)：还是利用用户和物品的交互矩阵,引入隐向量来表示物品和用户的向量

3.lgbm+lr模型：两个模型都够分别拿来进行任务，也可以利用两个模型组合，此时lgbm被视作是特征提取器(lgbm也可以被替换成其他模型)

4.因子分解机(Factor Machine,FM):引入特征的二阶交叉信息，同时计算公式通过化简可以降低复杂度至O(kn).(FM的改进为FFM模型,是在FM模型上引入了域的概念)

5.AutoRec：将自编码器和协同过滤的思想进行结合，通过MLP对用户进行自身学习，来输出对没有过评分的物品的评分

6.NeuCF(NeuMF):相当于是对协同过滤的神经网络化，通过神经网络来分别处理用户向量和物品向量，最后利用神经网络进行输出

7.ResNet

8.deepCrossing：所有离散特征embedding之后和dense特征concat后利用神经网络学习

9.wide&deep:利用wide部分的记忆能力和deep部分的泛化能力进行组合

10.DeepFM：wide&deep的变种，将wide部分改进为FM,从而实现了线性部分对二阶交叉特征的学习

11.deep&cross network(DCN):wide&deep的变种，将wide部分改为cross部分，从而避免了wide部分需要人工特征工程，并且能够高效的学习低维特征交叉。

12.xDeepFM:相当于DCN的改进(依旧属于wide&deep系列,虽然名字和deepFM很像但实质上关系不大),在DCN的cross部分引入域的概念(类似于FM->FFM),从而进行改进

13.NFM(Neural Factor Machine)：在获取embedding向量后类似于FM进行组合，从而提高隐藏层学习高阶非线性组合特征的能力

14.AFM(Attention Factorization Machine):对NFM的改进，对于在embedding向量组合的时候引入attention的概念进行组合

15-18：兴趣网络：DIN、DIEN、DSIN
