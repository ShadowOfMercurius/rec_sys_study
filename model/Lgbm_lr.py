# author:zh

#将lgbm作为一个特征提取，输送给后面的lr模型，将lgbm替换成其他模型也可以
from lightgbm.sklearn import LGBMClassifier,LGBMRegressor
from sklearn.linear_model import LogisticRegression


class LGBM_model:
    def __init__(self,task,params):
        if task == 'classifier':
            self.clf = LGBMClassifier(**params)
        elif task == 'regression':
            self.clf = LGBMRegressor(**params)

    def train(self,x,y):
        self.clf.fit(x,y)

    def predict(self,x,return_type):
        if return_type == 'category':
            return self.clf.predict(x)
        elif return_type == 'prob':
            return self.clf.predict_prob(x)
        else:
            return self.clf.predict_prob(x,pred_leaf=True)

    #....评估函数等根据需求添加

class LR_model:
    def __init__(self):
        self.lr_model = LogisticRegression()
    def train(self,x,y):
        self.lr_model.fit(x,y)
    def predict(self,x):
        return self.lr_model.predict_proba(x)


class LGBM_LR:
    def __init__(self,lgbm_params,task):
        self.lgbm_model = LGBM_model(task,lgbm_params)
        self.lr_model = LR_model()

    def train(self,x_lgb,y_lgb,x_lr,y_lr):
        self.lgbm_model.train(x_lgb,y_lgb)
        lr_input = self.lgbm_model.predict(x_lr)
        self.lr_model.train(lr_input,y_lr)

    def predict(self,x):
        lr_input = self.lgbm_model.predict(x)
        output = self.lr_model.predict(lr_input)


