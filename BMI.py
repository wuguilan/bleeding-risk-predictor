import pandas as pd

# 假设你有一个包含体重（weight）和身高（height）的 DataFrame

data = pd.read_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu_static_imputed.csv')
df = pd.DataFrame(data)
df['admissionheight'] = df['admissionheight'] / 100
# 计算 BMI
df['BMI'] = df['weight'] / (df['admissionheight'] ** 2)
df.to_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-static-BMI.csv', index=False)
# 查看结果
print(df[['patientunitstayid', 'weight', 'admissionheight', 'BMI']])
