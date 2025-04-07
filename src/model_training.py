#破坏模式和极限承载力模型的训练过程、超参数优化等，将训练好的模型存储好#
#===============================================================================
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import math
import joblib
import random
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score#,mean_absolute_error,mean_squared_error

#正常显示中文标签
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
#用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False 

###=================================破坏模式预测====================================
##破坏模式超参数优化函数
def hyper_mode(x,y):
    #定义参数取值范围
    model = XGBClassifier(objective='multi:softmax', num_class=3, booster='gbtree')

    #定义要搜索的超参数网格
    param_grid = {
        #学习率：
        'learning_rate': [0.01, 0.1, 0.2, 0.3], 
        #树的最大深度：
        'max_depth': [4, 5, 6], 
        #每棵树数据采样比例：                
        'subsample': [0.5, 0.8, 1.0],           
        #每棵树列采样比例：
        'colsample_bytree': [0.5, 0.8, 1.0],    
        #弱学习器的数量：
        'n_estimators': [100, 200, 300]         
       }

    #进行网格搜索和交叉验证
    #以'accuracy'为标准进行交叉验证
    clf = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=1, n_jobs=-1)
    clf = clf.fit(x, y)
    best_par=clf.best_params_
    
    #返回最优超参数组合
    return best_par

##定义破坏模式训练函数，存储训练好的模型
def train_classifier(df):
    #将预处理的数据划分特征矩阵和目标向量
    data_target_part = df['Deal failure mode']
    data_features_part = df[[x for x in df.columns if x != 'Deal failure mode']]

    #随机划分数据集，测试集大小为20%，
    x_train, x_test, y_train, y_test = train_test_split(
        data_features_part, data_target_part, test_size = 0.2, random_state=42) 
    
    #超参数优化（包含5个超参数）
    if os.path.exists("model\par_class.joblib"):
        print("正在加载已保存的最优超参数（破坏模式）...")
        best_par = joblib.load("model\par_class.joblib")
    else:
        print("正在进行网格搜索...")
        best_par = hyper_mode(x_train,y_train)
        joblib.dump(best_par, "model\par_class.joblib")
    
    #以最优超参数定义模型
    Model_class = xgb.XGBClassifier(**best_par,objective='multi:softmax',
                   num_class=3, booster='gbtree')
    
    #用训练集训练破坏模式预测模型并存储
    Model_class.fit(x_train, y_train)
    joblib.dump(Model_class, "model\model_class.pkl")
    print("破坏模式预测模型已保存！")


###=================================承载力预测====================================
##承载力超参数优化函数
def hyper_Pu(x,y):
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    # 定义要搜索的超参数网格
    param_grid = {
        #学习率
        'learning_rate': [0.1, 0.2, 0.3], 
        #树的最大深度
        'max_depth': [4, 5, 6],                 
        #每棵树数据采样比例
        'subsample': [0.5, 0.8, 1.0],           
        #每棵树列采样比例
        'colsample_bytree': [0.5, 0.8, 1.0],    
       }

    #进行网格搜索和交叉验证
    #以'MSE'为标准进行交叉验证
    clf = GridSearchCV(model, param_grid, cv=5,
                        scoring='neg_mean_squared_error',verbose=1, n_jobs=-1)
    clf = clf.fit(x, y)
    best_par=clf.best_params_

    #返回最优超参数组合
    return best_par

##定义有效粘结长度系数
def κ_judge(κ,Le,L):
    for i in range(len(κ)):
        if κ[i] >= 1:
            κ[i] = 1
        else:
            κ[i] = L[i]/Le[i]
    return κ

##定义承载力训练函数，存储训练好的模型
def train_regressor(df1):
    #将预处理的数据划分特征矩阵和目标向量
    df1=pd.DataFrame(df1)
    #特征矩阵
    x = df1.drop('Pu_half', axis=1) 
    #目标向量 
    y = df1['Pu_half']       

    #测试集大小为20%， 80%/20%分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42) 
    
    #对于XGBoost的原生接口，需要将训练数据转化为Dmatrix格式：
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    #得到纯数据驱动时的最优超参数组合
    if os.path.exists("model\par_reg.joblib"):
        print("正在加载已保存的最优超参数(承载力)...")
        best_par = joblib.load("model\par_reg.joblib")
    else:
        print("正在进行网格搜索...")
        best_par=hyper_Pu(x_train,y_train)
        joblib.dump(best_par, "model\par_reg.joblib")

    #引入先验物理知识自定义损失函数
    def custom_loss(y_pre,dtrain):    
        dlabel=dtrain.get_label()   
        data=dtrain.get_data()
        data=data.todense()  #稀疏矩阵转变为稠密矩阵
        fa=data[:,0]
        fa=np.array(fa).flatten()
        Ea=data[:,1]
        Ea=np.array(Ea).flatten()
        γa=data[:,2]
        γa=np.array(γa).flatten()
        ta=data[:,3]
        ta=np.array(ta).flatten()
        Ep=data[:,4]
        Ep=np.array(Ep).flatten()
        tp=data[:,5]
        tp=np.array(tp).flatten()
        bp=data[:,6]
        bp=np.array(bp).flatten()
        Es=data[:,7]
        Es=np.array(Es).flatten()
        ts=data[:,8]
        ts=np.array(ts).flatten()
        bs=data[:,9]
        bs=np.array(bs).flatten()
        L=data[:,10]
        L=np.array(L).flatten()
        Pu_half=dtrain.get_label()
        Pu_half=np.array(Pu_half).flatten()
    
        #使用修正的Xia Teng公式作为先验物理知识
        τf=0.8*fa
        Ga=Ea/(2*(1+γa))
        Gf_pre=31*pow((fa/Ga),0.56)*pow(ta,0.27)
        Le=np.pi/(np.sqrt(2*pow(τf,2)/(Ep*1000*tp*Gf_pre)))
        κ=L/Le
        κ=κ_judge(κ, Le, L)
        
        #计算残差项
        f=[]
        for i in range(len(fa)):
            f0=math.exp(-3.407)*math.exp(0.176*math.log(fa[i]))*math.exp(0.198/γa[i])*math.exp(48.537/Ep[i])*math.exp(-0.107/tp[i])*math.exp(6.492/bp[i])*math.exp(-0.023*ts[i])*math.exp(0.653*math.log(bs[i]))*math.exp(0.188/(L[i]/Le[i]))*math.exp(-0.087*math.log(Ea[i]*ta[i]))
            f.append(f0)
        f=np.array(f).flatten()
    
        Pu_for=f*κ*bp*np.sqrt(2*Gf_pre*Ep*1000*tp)/1000

        #自定义损失函数：
        #一阶导数：
        grad=λ1_*(y_pre-dlabel)+λ2_*(y_pre-Pu_for)
        #二阶导数，保持长度一致
        hess = 1.0 * np.ones_like(y_pre) 
        return grad,hess
    
    #得到纯数据驱动时的最优超参数组合
    if os.path.exists("model\λ1.joblib"):
        print("正在加载已保存的最优损失比例")
        best_λ1 = joblib.load("model\λ1.joblib")
        best_λ2 = joblib.load("model\λ2.joblib")
    else:
        #根据经验确定λ1和λ2的大致范围
        print("正在搜索最优损失比例...")
        λ1_range = np.arange(0.5, 1.5, 0.05)
        λ2_range = np.arange(0.1, 0.3, 0.01)
        num_samples = 50
        sample_λ1=[]
        sample_λ2=[]
        sample_r2=[]
        
        #对λ1和λ2进行随机搜索
        for _ in range(num_samples):
            λ1 = random.choice(λ1_range)
            sample_λ1.append(λ1)
            λ1_=λ1*np.ones_like(y_train)

            λ2 = random.choice(λ2_range)
            sample_λ2.append(λ2)
            λ2_=λ2*np.ones_like(y_train)

            model= xgb.train(best_par, dtrain, num_boost_round=100, obj=custom_loss)  
            #该组合下使用测试集进行预测
            test_predict = model.predict(dtest)   
            #R^2 (决定系数)
            r2 = r2_score(y_test, test_predict)   
            sample_r2.append(r2)
    
        #找到r2列表中最大元素对应的索引
        max_index = sample_r2.index(max(sample_r2))

        #根据索引找到最优的λ1和λ2
        best_λ1 = sample_λ1[max_index]
        best_λ2 = sample_λ2[max_index]
        joblib.dump(best_λ1, "model\λ1.joblib")
        joblib.dump(best_λ2, "model\λ2.joblib") 
    
    #找到最优的损失比例并存储承载力预测模型
    λ1_=best_λ1*np.ones_like(y_train)
    λ2_=best_λ2*np.ones_like(y_train)
    Model_reg= xgb.train(best_par, dtrain, num_boost_round=100, obj=custom_loss) 
    joblib.dump(Model_reg, "model\model_reg.pkl")
    print("承载力预测模型已保存！") 
