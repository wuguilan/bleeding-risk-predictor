import pandas as pd

# 读取数据
file_path = "D:/所有工作进展/毕业学位论文相关/数据/时间序列lab.csv"
df = pd.read_csv(file_path)

# 删除完全为空的列
df_cleaned = df.dropna(axis=1, how='all')

# 确保 `Timestamp` 不是空值，并转换为 `datetime`
df_cleaned = df_cleaned.dropna(subset=['Timestamp'])
df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'])

# 排序数据
df_cleaned = df_cleaned.sort_values(['patientunitstayid', 'Timestamp'])

# 去掉重复的 Timestamp，保留每个 patientunitstayid 下的唯一时间戳
df_cleaned = df_cleaned.drop_duplicates(subset=['patientunitstayid', 'Timestamp'])

# **去掉 MultiIndex，单独把 `Timestamp` 设为索引**
# 使用 groupby 分组后，再对每个分组执行重采样
df_resampled = df_cleaned.groupby('patientunitstayid').apply(
    lambda group: group.set_index('Timestamp').resample('H').pad().reset_index()
).reset_index(drop=True)

# 保存处理后的数据
df_resampled.to_csv("processed_timeseries.csv", index=False)
print("数据处理完成，已保存为 processed_timeseries.csv")
