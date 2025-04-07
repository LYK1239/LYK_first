# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:29:18 2025
@author: LiuYikang
"""
#对训练数据进行预处理，将处理好的数据存储好便于后续的调用#
#======================================================================
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict

##破坏模式编码：把所有的相同类别的特征编码为同一个值
    #0:C-胶层内聚失效； 1:SA-钢与胶层界面破坏； 2:D-CFRP层间剥离破坏
def get_mapfunction(x):
    mapp = dict(zip(x.unique().tolist(),
                    range(len(x.unique().tolist()))))
    def mapfunction(y):
        if y in mapp:
            return mapp[y]
        else:
            return -1
    return mapfunction

##异常值处理：孤立森林算法
from sklearn.ensemble import IsolationForest
def ISO(X):
    #X是包含11个输入和1个输出（破坏模式/极限承载力）的Dataframe表格
    #设置孤立森林算法
    iso_forest = IsolationForest(
                 n_estimators=100, 
                 max_samples='auto', 
                 contamination='auto', 
                 max_features=1.0, 
                 bootstrap=False, 
                  n_jobs=None, 
                 random_state=42, 
                  verbose=0, 
                  warm_start=False)
    #训练孤立森林模型
    iso_forest.fit(X)
    #预测每个样本是否为异常值
    outliers = iso_forest.predict(X)
    #标记异常值
    X['outlier'] = outliers
    #-1 表示异常值，1 表示正常值
    #过滤掉异常值
    filtered_data = X[X['outlier'] == 1].drop(columns='outlier')
    return filtered_data

##数据预处理函数
def data_preprocessing(filename):
    #导入原始数据（未处理）
    df = pd.read_excel(filename,sheet_name='Raw_data')
    
    # 记录数字特征和非数字特征
    category_features = [x for x in df.columns 
                         if df[x].dtype != np.float64 and df[x].dtype != np.int64]
    
    #调用get_mapfunction函数将所有非数字特征（破坏模式）转换为数字特征
    for i in category_features:
        df[i] = df[i].apply(get_mapfunction(df[i]))

    #从原始数据中分离出输入特征和输出特征
    #前11列为输入特征：
    X = df.iloc[:, :11]  
    #分类标签
    y_class = df['Deal failure mode']  
    #回归标签
    y_reg = df['Pu_half']
    
    #转换为DataFrame格式： 
    X=pd.DataFrame(X)
    y_class=pd.DataFrame(y_class)
    y_reg=pd.DataFrame(y_reg)

    #表格合并：分别合并用于破坏模式预测的表格和用于极限承载力预测的表格
    df_mode = pd.concat([X, y_class], axis=1)
    df_Pu = pd.concat([X, y_reg], axis=1)
    
    #分别进行异常值处理
    df_mode_trusted=ISO(df_mode)
    df_Pu_trusted=ISO(df_Pu)
    
    #将与处理好的数据存储到joblib文件中，便于后续的调用
    if os.path.exists("data\Data_reg_selected.joblib"):
         print("Data_reg_selected already exists!")
    else:
       joblib.dump(df_Pu_trusted, "data\Data_reg_selected.joblib")
    return df_mode_trusted,df_Pu_trusted

##对破坏模式数据集进行特征选择
def feature_select(df):
    #划分特征矩阵和目标向量
    #特征矩阵：
    X = df.drop('Deal failure mode', axis=1)
    #目标向量： 
    y = df['Deal failure mode']               

    #设置随机种子以便结果可复现
    np.random.seed(42)

    #进行多次RFECV以获得稳定的特征数量
    #运行次数：
    num_runs = 10  
    #存储每次迭代的特征集合：
    feature_sets = defaultdict(list)  
    #存储每次迭代的交叉验证得分
    scores = defaultdict(list)
    rank = defaultdict(list)

    for _ in range(num_runs):
        #以决策树为测试模型：
        estimator = DecisionTreeClassifier() 
        rfecv = RFECV(estimator, cv=10, scoring='accuracy')
        rfecv = rfecv.fit(X, y)
        #存储选中的特征集合：
        feature_sets[rfecv.n_features_].append(list(X.columns[rfecv.support_]))
        #存储交叉验证得分：
        scores[rfecv.n_features_].append(rfecv.cv_results_['mean_test_score'])
        #存储排名：
        feature_importance = pd.Series(rfecv.ranking_, index=X.columns)
        rank[rfecv.n_features_].append(feature_importance)
        #计算特征数量的中位数
        optimal_num_features = int(np.min(list(feature_sets.keys())))

        #选中的特征集合及其对应的平均交叉验证得分
        for num_features, feature_list in feature_sets.items():
            if num_features == optimal_num_features:
                opt_scores = scores[num_features]
                rank_features = rank[num_features]
                #可以选择输出
                '''print(f"\nFeatures with {num_features} columns:")
                print(feature_list)  
                print(f"Average cross-validation scores: {opt_scores}")
                print(f"Ranking of features:{rank_features}")'''
        
        #feature_list是一个列表的列表，取第一个列表
        X_selected=df[feature_list[0]] 
        df_mode_selected= pd.concat([X_selected, y], axis=1)
    
    #将与处理好的数据存储到joblib文件中，便于后续的调用
    if os.path.exists("data\Data_class_selected.joblib"):
        print("Data_class_selected already exists!")
    else:
        joblib.dump(df_mode_selected, "data\Data_class_selected.joblib")
    return df_mode_selected