#定义GUI界面类，设置各功能按钮，实现CFRP-钢界面黏结性能预测功能的可视化#
#===============================================================================
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QLabel, 
    QLineEdit, QPushButton, QMessageBox, 
    QFormLayout,QHBoxLayout,QFileDialog
)
from PyQt5.QtCore import QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QIcon
from src.model_predicting import Predictor

class CFRPApp(QWidget):
    """
    定义主界面类
    """
    def __init__(self):
        super().__init__()
        #初始化预测器
        self.predictor = Predictor()  
        #初始化界面
        self.initUI() 
        #设置动画效果
        self.setupAnimation()  

        #定义特征列名（与训练数据一致）
        self.feature_columns = [
            'fa(MPa)', 'Ea(MPa)', 'υa', 'ta(mm)', 'Ep(GPa)', 
            'tp(mm)', 'bp(mm)', 'Es(GPa)', 'ts(mm)', 'bs(mm)', 'L(mm)'
        ]

        #用于存储预测结果
        self.results_df = None  

        #加载特征选择信息（破坏模式预测模型）
        self.selected_features = joblib.load("data/Data_class_selected.joblib").columns.tolist()

    def initUI(self):
        """
        初始化界面
        """
        self.setWindowTitle("CFRP-钢界面性能预测系统")
        
        #设置窗口图标
        self.setWindowIcon(QIcon("icon.png"))  
        
        #设置窗口大小和位置
        self.setGeometry(100, 100, 400, 300)  

        #使用 QFormLayout 布局
        layout = QFormLayout()

        #输入框标签和提示
        self.labels = [
            "结构黏胶抗拉强度 (MPa)", "结构黏胶弹性模量 (MPa)", "结构黏胶泊松比", "胶层厚度 (mm)",
            "CFRP弹性模量 (GPa)", "CFRP厚度 (mm)", "CFRP宽度 (mm)", "钢板弹性模量 (GPa)",
            "钢板厚度 (mm)", "钢板宽度 (mm)", "黏结长度 (mm)"
        ]

        #收集输入框中的数据
        self.entries = []
        for label in self.labels:
            entry = QLineEdit(self)
            #设置输入框提示：
            entry.setPlaceholderText(f"请输入{label}")  
            layout.addRow(QLabel(label), entry)
            self.entries.append(entry)

        #设置布局
        self.setLayout(layout)
            
        #设置样式
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: "微软雅黑";
                font-size: 14px;
            }
            QLabel {
                color: #333333;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        #按钮区域
        btn_layout = QHBoxLayout()

        #预测按钮
        btn = QPushButton("预测", self)
        btn.clicked.connect(self.on_predict)
        layout.addRow(btn)

        #帮助文档按钮
        self.btn_help = QPushButton("使用说明", self)
        self.btn_help.clicked.connect(self.show_help)
        layout.addRow(self.btn_help)

        #文件导入按钮
        self.btn_import = QPushButton("导入文件", self)
        self.btn_import.clicked.connect(self.import_data)
        btn_layout.addWidget(self.btn_import)

        #结果导出按钮
        self.btn_export = QPushButton("导出结果", self)
        self.btn_export.clicked.connect(self.export_data)
        btn_layout.addWidget(self.btn_export)

        layout.addRow(btn_layout)

    def setupAnimation(self):
        """
        设置窗口弹出动画
        """
        self.animation = QPropertyAnimation(self, b"geometry")
        #动画时长：500ms：
        self.animation.setDuration(500)  
        #初始大小：
        self.animation.setStartValue(QRect(100, 100, 0, 0))  
        #最终大小：
        self.animation.setEndValue(QRect(100, 100, 400, 300))  
        #动画效果：
        self.animation.setEasingCurve(QEasingCurve.OutBack)  
        self.animation.start()

    def show_help(self):
        '''
        设置帮助文档
        '''
        help_text = """
            CFRP-钢界面性能预测系统使用说明：
            1. 本软件基于CFRP-钢界面拉伸剪切试验样本，使用XGBoost算法对CFRP-钢界面破坏模式和极限承载力进行预测，
                适用于此问题相近场景的预测；
            2. 本软件支持手动输入或文件导入(CSV/Excel)，表头须严格按照以下顺序:
                fa(MPa),Ea(MPa),υa,ta(mm),Ep(GPa),tp(mm),
                bp(mm),Es(GPa),ts(mm),bs(mm),L(mm)
            3. 数据导入成功后点击预测按钮即可获得CFRP-钢界面的破坏模式和极限承载力预测值,
                需注意，此结果为机器学习预测值，与真实值可能存在偏差;
            4. 预测完成后，可以将结果导出为CSV文件
            """
        QMessageBox.information(self, "帮助文档", help_text)

    def import_data(self):
        """
        导入CSV/Excel文件
        """
        file_path, _ = QFileDialog.getOpenFileName(
        self, "选择文件", "", "数据文件 (*.xlsx *.csv)")
        
        if not file_path:
             return
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        
            #验证列名是否匹配
            expected_cols = self.feature_columns
            if not all(col in df.columns for col in expected_cols):
                QMessageBox.critical(self, "错误", "文件列名与要求不一致！")
                return

            #严格验证列名
            missing_cols = set(expected_cols) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_cols)
        
            if missing_cols:
                QMessageBox.critical(
                    self, "列名不匹配",
                    f"缺少必要列: {', '.join(missing_cols)}\n"
                    f"需要的列名: {', '.join(expected_cols)}"
                )
                return
        
            if extra_cols:
                reply = QMessageBox.question(
                    self, "发现额外列",
                    f"检测到额外列: {', '.join(extra_cols)}\n"
                    "是否继续使用匹配的列？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        
            #只保留需要的列（按原始顺序）
            df = df[expected_cols]

            #清空现有输入
            for entry in self.entries:
                entry.clear()
        
            #填入第一行数据（示例）
            first_row = df.iloc[0]
            for col, entry in zip(expected_cols, self.entries):
                entry.setText(str(first_row[col]))
            
            #存储完整数据供批量预测
            self.imported_data = df
            QMessageBox.information(self, "成功", f"成功加载 {len(df)} 条数据！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件读取失败:\n{str(e)}")

    def on_predict(self):
        """
        预测按钮点击事件
        """
        try:
            # 优先使用导入的数据
            if self.imported_data is not None:
            # 批量预测逻辑
                all_features_df = self.imported_data
            else:
                # 单样本预测逻辑（原有代码）
                # 1. 获取所有输入值
                all_features = {}
                for col_name, entry in zip(self.feature_columns, self.entries):
                    value = entry.text().strip()
                    if not value:
                        raise ValueError("输入不能为空")
                    all_features[col_name] = [float(value)]
            
                    # 转换为完整DataFrame
                    all_features_df = pd.DataFrame(all_features)
            
            #print("所有输入特征:\n", all_features_df)
            # 2. 分别处理两个预测模型
            # 2.1 承载力预测（使用全部特征）
            features_reg = xgb.DMatrix(all_features_df)
            strengths = self.predictor.predict_Pu(features_reg)
            strengths = pd.DataFrame(strengths)

            # 2.2 破坏模式预测（使用特征选择后的特征）
            #print(self.selected_features)
            selected_features0=self.selected_features
            selected_features=selected_features0[0:7]
            features_class = all_features_df[selected_features]
            #print("用于破坏模式预测的特征:\n", features_class)
            modes = self.predictor.predict_mode(features_class) 

            self.results_df = all_features_df.copy()
            self.results_df['预测破坏模式'] = [self._map_mode(m) for m in modes]
            self.results_df['预测承载力(kN)'] = np.round(strengths, 2)
            # 3. 显示结果
            self.show_result_table()
        

        except ValueError as e:
            # 错误提示
            error_message = "输入错误！请确保：\n"
            error_message += "1. 所有字段都已填写\n"
            error_message += "2. 输入值为有效的数字\n"
            error_message += "3. 单位与提示一致（如MPa, GPa, mm）"
            QMessageBox.critical(self, "输入错误", error_message)
        except Exception as e:
            QMessageBox.critical(self, "系统错误", f"预测过程中发生错误:\n{str(e)}")

    def _map_mode(self, mode_code):
        """
        将破坏模式的数字编码转换为文字说明
        """
        mode_mapping = {
            0: "C: 胶层内聚失效",
            1: "SA: 钢-胶层界面失效",
            2: "D: CFRP层间剥离失效"
            }
        return mode_mapping.get(mode_code, "未知模式")

    def show_result_table(self):
        """
        结果展示窗口
        """
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
        
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("批量预测结果")
        result_dialog.resize(800, 600)
        
        table = QTableWidget()
        table.setRowCount(len(self.results_df))
        table.setColumnCount(len(self.results_df.columns))
        
        #设置表头
        table.setHorizontalHeaderLabels(self.results_df.columns.tolist())
        
        #填充数据
        for i, row in self.results_df.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                table.setItem(i, j, item)
        
        layout = QVBoxLayout()
        layout.addWidget(table)
        result_dialog.setLayout(layout)
        result_dialog.exec_()

    def export_data(self):
        """
        导出预测结果
        """
        if self.results_df is None:
            QMessageBox.warning(self, "警告", "没有可导出的结果！")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "CSV文件 (*.csv)")
        if not save_path:
            return
    
        #self.results_df是存储结果的DataFrame
        try:
            self.results_df.to_csv(save_path, index=False)
            QMessageBox.information(self, "成功", "结果导出成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")