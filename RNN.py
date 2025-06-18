import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# **1. 读取数据**
static_data_file = r"D:\所有工作进展\毕业学位论文相关\数据\sepsis5.csv"
high_freq_data_file = r"D:\所有工作进展\毕业学位论文相关\数据\时间序列 vital.csv"
low_freq_data_file = r"processed_timeseries.csv"

# 读取静态数据
static_df = pd.read_csv(static_data_file)

# 读取高频和低频动态数据
high_freq_df = pd.read_csv(high_freq_data_file)  # 高频数据
low_freq_df = pd.read_csv(low_freq_data_file)    # 低频数据

# **2. 数据预处理**
# 提取静态数据列和目标变量（例如 VTE）
static_cols = ['age', 'gender']  # 示例静态特征
target_col = 'vte'  # 假设目标变量是 VTE（静脉血栓栓塞）

# 提取静态数据和目标变量
X_static = static_df[static_cols].values  # 静态数据
y = static_df[target_col].values  # 目标变量

# **3. 合并高频和低频数据**
# 假设高频数据每小时更新，低频数据每4小时更新
# 设置时间列进行合并（如果有时间戳，可以按时间对齐数据）
high_freq_df['Timestamp'] = pd.to_datetime(high_freq_df['Timestamp'])
low_freq_df['Timestamp'] = pd.to_datetime(low_freq_df['Timestamp'])

# 进行重采样：将低频数据重采样为高频数据的时间戳（例如，填充每小时数据）
low_freq_resampled = low_freq_df.set_index('Timestamp').resample('H').pad().reset_index()

# **4. 合并两类数据**
# 假设高频数据和低频数据都有时间戳，我们可以合并它们
merged_df = pd.merge(high_freq_df, low_freq_resampled, on='Timestamp', suffixes=('_high', '_low'))

# **5. 数据重塑：准备动态数据**
# 假设每个时间步下有不同时间间隔的数据列（例如，心率为高频数据，实验室检查结果为低频数据）

# 提取动态数据列
high_freq_cols = [col for col in merged_df.columns if 'high' in col]
low_freq_cols = [col for col in merged_df.columns if 'low' in col]

# 合并高频和低频特征
dynamic_cols = high_freq_cols + low_freq_cols  # 将高频数据和低频数据列合并

# 获取动态数据（每个样本的时间步数据）
dynamic_data = merged_df[dynamic_cols].values

# 假设有 24 个时间步，且每个时间步下有多个特征（包括高频和低频）
num_time_steps = 24  # 时间步数量
num_features = len(dynamic_cols)  # 每个时间步的特征数量（高频 + 低频）

# 重新组织动态数据形状：样本数 × 时间步 × 特征数
X_dynamic = dynamic_data.reshape(-1, num_time_steps, num_features)

# **6. 划分训练集和测试集**
# 使用 tensorflow 来划分数据集
dataset_size = len(X_static)
train_size = int(dataset_size * 0.8)

# 划分静态数据
X_train_static = X_static[:train_size]
X_test_static = X_static[train_size:]

# 划分动态数据
X_train_dynamic = X_dynamic[:train_size]
X_test_dynamic = X_dynamic[train_size:]

# 划分目标变量
y_train = y[:train_size]
y_test = y[train_size:]

# **7. 构建GRU模型 + 注意力机制**
input_dynamic = Input(shape=(num_time_steps, num_features))  # 动态数据输入
input_static = Input(shape=(len(static_cols)))  # 静态数据输入

# GRU 网络处理时间序列数据
gru_out = GRU(64, return_sequences=True)(input_dynamic)  # 返回序列，以便传入注意力机制
gru_out = Dropout(0.3)(gru_out)

# **注意力机制**
attention = Attention()([gru_out, gru_out])  # 自注意力机制，计算GRU输出的加权版本
attention = LayerNormalization()(attention)  # 进行层归一化

# 对注意力加权后的输出进行池化（如 GlobalAveragePooling）
attention_out = np.mean(attention, axis=1)  # 或使用 GlobalMaxPooling1D 等

# 将静态数据与 GRU 和注意力机制的输出拼接
x = Concatenate()([attention_out, input_static])

# 全连接层输出
output = Dense(1, activation='sigmoid')(x)  # 假设是二分类任务，输出是否发生VTE

# 构建模型
model = Model(inputs=[input_dynamic, input_static], outputs=output)

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# **8. 训练模型**
model.fit([X_train_dynamic, X_train_static], y_train, epochs=10, batch_size=32, validation_data=([X_test_dynamic, X_test_static], y_test))

# **9. 模型评估**
test_loss, test_acc = model.evaluate([X_test_dynamic, X_test_static], y_test)
print(f'测试集损失: {test_loss}, 测试集准确率: {test_acc}')
