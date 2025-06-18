import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# 路径设置
data_path = 'D:\\所有工作进展\\毕业学位论文相关\\数据\\filtered_static_data_50percent.csv'
output_path = 'D:\\所有工作进展\\毕业学位论文相关\\数据\\'

# 读取数据
df = pd.read_csv(data_path)

# 提取 ID 和标签
id_col = 'stay_id'
label_col = 'vte'

# 拆分列
id_vals = df[id_col]
labels = df[label_col]
features = df.drop([id_col, label_col], axis=1)

# 记录最佳结果
best_auc = 0
best_df = None

# MICE 插补器设置
for i in range(5):
    print(f"\n🔁 第 {i+1} 个插补数据集")

    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=i, sample_posterior=True)
    imputed_array = imputer.fit_transform(features)
    imputed_df = pd.DataFrame(imputed_array, columns=features.columns)

    # 恢复 ID 和标签列
    imputed_df[id_col] = id_vals.values
    imputed_df[label_col] = labels.values
    imputed_df = imputed_df[[id_col] + list(features.columns) + [label_col]]

    # 拆分训练/验证
    X = imputed_df.drop([id_col, label_col], axis=1)
    y = imputed_df[label_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 分类器训练
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)

    print(f"✅ 第 {i+1} 个插补数据集 AUC：{auc:.4f}")

    # 保存最佳
    if auc > best_auc:
        best_auc = auc
        best_df = imputed_df.copy()
        best_index = i + 1

# 保存最佳插补数据
output_path = f'{output_base}best_{best_index}_auc_{best_auc:.4f}.csv'
best_df.to_csv(output_path, index=False)

print(f"\n🏆 最佳插补数据集为第 {best_index} 个，AUC = {best_auc:.4f}")
print(f"📁 已保存到：{output_path}")
