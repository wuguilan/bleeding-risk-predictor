import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import pandas as pd


# 定义GRU模型
def create_gru_model(input_size, hidden_size=40):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(hidden_size, input_shape=(None, input_size), activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# 数据预处理
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler
scaled_test_data = scaler.transform(test_data)

# 交叉验证训练
def train_model(X, y, input_size, num_epochs=5, batch_size=32):
    skf = StratifiedKFold(n_splits=10)
    val_predictions = []
    test_predictions = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练数据预处理
        X_train, scaler = preprocess_data(X_train)
        X_val = scaler.transform(X_val)

        # 将数据格式转换为符合LSTM要求的形状 (samples, time_steps, features)
        X_train = np.expand_dims(X_train, axis=1)  # 加入时间步维度
        X_val = np.expand_dims(X_val, axis=1)

        # 创建和训练GRU模型
        model = create_gru_model(input_size=input_size, hidden_size=40)

        # 训练模型
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

        # 验证集预测
        val_pred = model.predict(X_val)
        val_predictions.append(val_pred)

    # 对每个模型的预测进行平均
    avg_val_pred = np.mean(val_predictions, axis=0)

    # 输出最终的预测结果
    final_pred = (avg_val_pred > 0.5).astype(int)

    return final_pred, avg_val_pred


# 假设你已经加载了数据
# 假设 X 是你的特征数据，y 是对应的标签
# X, y = load_your_data()

# 对数据的每个输入特征维度进行标准化
input_size = X.shape[1]  # 特征数量

# 执行交叉验证和训练
final_pred, avg_val_pred = train_model(X, y, input_size, num_epochs=5)

# 计算准确率
accuracy = accuracy_score(y, final_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 计算二进制交叉熵损失
logloss = log_loss(y, avg_val_pred)
print(f'Log Loss: {logloss:.4f}')
