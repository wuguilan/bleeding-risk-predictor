import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# 加载你的数据（如果是CSV）
df = pd.read_csv(r"C:\Users\14340\Desktop\无标题.csv")  # ← 如果你是CSV数据

# 示例：你已有 DataFrame df，包含如下字段
id_cols = ['subject_id', 'hadm_id', 'stay_id']
non_impute_cols = ['icu_intime', 'icu_outtime', 'vte', 'gender']  # 不参与插补的列
impute_cols = [col for col in df.columns if col not in id_cols + non_impute_cols]

# 将性别转为数值型（如果你后续希望插补它也可以）
df['gender'] = df['gender'].map({'M': 1, 'F': 0})

# 初始化多重插补器
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)

# 插补
imputed_array = imputer.fit_transform(df[impute_cols])

# 转回 DataFrame
df_imputed = pd.DataFrame(imputed_array, columns=impute_cols)

# 合并回完整数据
df_final = pd.concat([df[id_cols + non_impute_cols].reset_index(drop=True), df_imputed], axis=1)

# 检查是否还有缺失值
print(df_final.isnull().sum())

df_final.to_csv('imputed_result.csv', index=False)

