import pandas as pd

# 读取数据
df = pd.read_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu_dyn_50percent.csv')

# 排序
df = df.sort_values(by=['patientunitstayid', 'time_window'])

# 保存 patientunitstayid 和其他非数值列
id_columns = df[['patientunitstayid']].copy()

# 对除 ID 外的所有列做前向填充
df_numeric = df.drop(columns=['patientunitstayid'])
df_filled = df_numeric.groupby(df['patientunitstayid']).ffill()

# 把 ID 加回去
df_ffill = pd.concat([id_columns, df_filled], axis=1)

# 打印看看结果
print(df_ffill.head())
df_ffill.to_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-dyn-ffill.csv', index=False)
