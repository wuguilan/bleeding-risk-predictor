import pandas as pd

# 假设你已经有了静态数据和动态数据的 DataFrame
static_df = pd.read_csv(r"D:\所有工作进展\毕业学位论文相关\数据\eicu_static_imputed.csv")  # 静态数据
dynamic_df = pd.read_csv(r"D:\所有工作进展\毕业学位论文相关\数据\eicu-dyn-imputed.csv")  # 动态数据

# 将静态数据表中的 VTE 标签添加到动态数据表中
# 静态数据按 subject_id 进行合并，VTE 标签会同步到动态数据中
merged_df = dynamic_df.merge(static_df[['patientunitstayid', 'vte']], on='patientunitstayid', how='left')
merged_df.to_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-dyn-merged.csv', index=False)
# 查看合并后的数据
print(merged_df.head())
