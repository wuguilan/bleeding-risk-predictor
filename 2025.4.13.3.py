import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, Dropout

# 动态输入：形状为 (时间步数, 动态特征数)
input_dynamic = Input(shape=(3, 26), name='dynamic_input')  # ⬅ 这里 3 是时间步（8小时×3），26 是动态特征个数
x_dynamic = GRU(64, return_sequences=False)(input_dynamic)  # 输出一个固定长度的向量
x_dynamic = Dropout(0.3)(x_dynamic)

# 静态输入：形状为 (静态特征数,)
input_static = Input(shape=(15,), name='static_input')  # ⬅ 这里 15 是你静态特征个数，改成你自己的
x_static = Dense(32, activation='relu')(input_static)
x_static = Dropout(0.3)(x_static)

# 融合两部分
x = Concatenate()([x_dynamic, x_static])
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)  # ⬅ 二分类问题用 sigmoid

# 构建模型
model = Model(inputs=[input_dynamic, input_static], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# 打印结构
model.summary()
# X_dynamic：形状为 (样本数, 时间步, 动态特征数)
# X_static：形状为 (样本数, 静态特征数)
# y：标签，形状为 (样本数,)

model.fit(
    x = {'dynamic_input': (80637,3,24), 'static_input': (26879,16)},
    y = (1,),
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)
