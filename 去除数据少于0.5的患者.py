import pandas as pd

# 路径设置
dynamic_data_path = 'D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-dyn.csv'
static_data_path = 'D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-sta.csv'

# 加载动态数据
dynamic_df = pd.read_csv(dynamic_data_path)

# 不参与缺失判断的列
exclude_cols = ['patientunitstayid', 'time_window']  # 如果没有 time_window，就只排除 stay_id
cols_to_check = [col for col in dynamic_df.columns if col not in exclude_cols]

# 计算每个 stay_id 的非缺失比例
def calculate_non_missing_ratio(group):
    total_cells = group[cols_to_check].shape[0] * len(cols_to_check)
    non_missing_cells = group[cols_to_check].notna().sum().sum()
    return non_missing_cells / total_cells if total_cells > 0 else 0

# 分组计算
non_missing_ratio_per_patient = dynamic_df.groupby('patientunitstayid').apply(calculate_non_missing_ratio)

# 保留非缺失比例 >= 60% 的患者
valid_stay_ids = non_missing_ratio_per_patient[non_missing_ratio_per_patient >= 0.5].index

# 筛选动态数据
filtered_dynamic_df = dynamic_df[dynamic_df['patientunitstayid'].isin(valid_stay_ids)]
filtered_dynamic_df.to_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu_dyn_50percent.csv', index=False)

# 同步筛选静态数据
static_df = pd.read_csv(static_data_path)
filtered_static_df = static_df[static_df['patientunitstayid'].isin(valid_stay_ids)]
filtered_static_df.to_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu_static_data_50percent.csv', index=False)

print(f"动态数据清洗后保留 {filtered_dynamic_df['patientunitstayid'].nunique()} 位患者")
print(f"静态数据清洗后保留 {filtered_static_df['patientunitstayid'].nunique()} 位患者")
