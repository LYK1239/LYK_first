import sys
import os
import joblib
from PyQt5.QtWidgets import QApplication
from src.gui import CFRPApp
from src.preprocess import data_preprocessing,feature_select
from src.model_training import train_classifier,train_regressor

if os.path.exists("model\model_class.pkl") and os.path.exists("model\model_reg.pkl"):
    print("模型已准备！")

else:
    #加载数据集
    if os.path.exists("data\Data_reg_selected.joblib"):
        print("正在加载已处理好的数据集...")
        df_class_selected= joblib.load("data\Data_class_selected.joblib")
        df_reg_trusted=joblib.load("data\Data_reg_selected.joblib")
    else: 
        print("正在处理数据集...")
        df_class_trusted,df_reg_trusted=data_preprocessing("data\Data.xlsx")
        df_class_selected=feature_select(df_class_trusted)
    #训练模型并保存
    train_classifier(df_class_selected)
    train_regressor(df_reg_trusted)

if __name__ == "__main__":
    # 启动应用程序
    app = QApplication(sys.argv)
    window = CFRPApp()
    window.show()
    sys.exit(app.exec_())