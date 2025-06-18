import pandas as pd
from statsmodels.imputation.mice import MICEData

# 读取原始数据
df = pd.read_csv("D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-dyn-ffill.csv")

# 选择需要插补的数值列（非字符串、ID列）
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# 如果有不希望参与插补的列（如ID），可以排除
exclude_cols = ['patientunitstayid', 'time_window']
impute_cols = [col for col in numeric_cols if col not in exclude_cols]

# 创建一个插补数据对象（只保留需要插补的列）
df_for_impute = df[impute_cols].copy()
mice_data = MICEData(df_for_impute)

# 设置插补轮数（每组插补迭代几次）
mice_data.update_all(10)  # 每组进行10次更新，使其收敛

# 生成多个完整数据集
imputed_datasets = []
n_imputations = 5

for i in range(n_imputations):
    mice_data.update_all(10)  # 每组插补前再迭代几次，确保生成不同版本
    completed = mice_data.data.copy()

    # 拼回原始的ID列
    completed['patientunitstayid'] = df['patientunitstayid'].values
    completed['time_window'] = df['time_window'].values

    imputed_datasets.append(completed)

    # 可保存到文件（可选）
    completed.to_csv(f"D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-dynfill-imputed.csv", index=False)
