import pandas as pd

# 读取数据
file_path = r"D:\Program data\PythonProject1\merged_data.csv"  # 请替换为你的实际路径
df = pd.read_csv(file_path)

# 创建时间段的分箱
bins = [0,12, 24]  # 定义时间段
labels = ["0-12", "12-24"]  # 给时间段命名

# 把 Timestamp 归类到时间段
df["TimePeriod"] = pd.cut(df["Timestamp"], bins=bins, labels=labels, right=False)

# 按照 patientunitstayid 和 TimePeriod 进行分组，并合并数据
df_combined = df.groupby(['patientunitstayid', 'TimePeriod'], as_index=False).max()

# 选择合适的本地路径保存文件
output_path = r"D:\Program data\PythonProject1\merged_data4.csv"
df_combined.to_csv(output_path, index=False)

print(f"文件已保存到 {output_path}")
