#定义预测类，便于两种模型的调用和预测#
#======================================================================
# -*- coding: utf-8 -*-
import joblib

class Predictor:
    def __init__(self):
        # 加载模型
        self.class_model = joblib.load("model\model_class.pkl")
        self.reg_model = joblib.load("model\model_reg.pkl")

    def predict_mode(self, features: list) -> tuple:
        mode = self.class_model.predict(features)  # 分类预测
        return mode
    
    def predict_Pu(self, features: list) -> tuple:
        strength = self.reg_model.predict(features)  # 回归预测
        return strength # 承载力保留两位小数