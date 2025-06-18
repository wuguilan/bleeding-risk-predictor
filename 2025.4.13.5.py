import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, Dropout
import tensorflow as tf
from tensorflow.keras import regularizers

# 1. 加载动态特征数据
df_dynamic = pd.read_csv(r"D:\所有工作进展\毕业学位论文相关\数据\填补后的数据.csv")
print(f"Shape of df_dynamic: {df_dynamic.shape}")

# 2. 重塑动态特征数据
time_steps = 3
features_per_time_step = 22
X_dynamic = df_dynamic.values.reshape(-1, time_steps, features_per_time_step)
print(f"Reshaped X_dynamic: {X_dynamic.shape}")

# 3. 加载静态特征数据
df_static = pd.read_csv(r"D:\所有工作进展\毕业学位论文相关\数据\imputed_result.csv")
y = df_static["vte"].values
X_static = df_static.drop(columns=["vte"]).values
print(f"Static features shape: {X_static.shape}")

# 4. 构建模型
input_dynamic = Input(shape=(time_steps, features_per_time_step), name='dynamic_input')
x_dynamic = GRU(64, return_sequences=False)(input_dynamic)
x_dynamic = Dropout(0.3)(x_dynamic)

input_static = Input(shape=(X_static.shape[1],), name='static_input')
x_static = Dense(32, activation='relu')(input_static)
x_static = Dropout(0.3)(x_static)

x = Concatenate()([x_dynamic, x_static])
x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input_dynamic, input_static], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# 5. 训练模型（不分割测试集，使用全部数据训练）
history = model.fit(
    x={'dynamic_input': X_dynamic, 'static_input': X_static},
    y=y,
    epochs=20,
    batch_size=32,
    validation_split=0.2  # 仍然保留20%作为验证集
)

# 6. 可视化训练过程
plt.figure(figsize=(12, 5))

# 训练和验证的损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 训练和验证的AUC曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Training and Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()

# 7. 输出最终训练结果
final_train_auc = history.history['auc'][-1]
final_val_auc = history.history['val_auc'][-1]
print(f"\nFinal Training AUC: {final_train_auc:.4f}")
print(f"Final Validation AUC: {final_val_auc:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# 模拟历史数据（实际使用时替换为您的history.history数据）
history = {
    'loss': np.linspace(0.55, 0.25, 20),
    'val_loss': np.linspace(0.40, 0.30, 20),
    'auc': np.linspace(0.5, 0.7, 20),
    'val_auc': np.linspace(0.525, 0.675, 20)
}

# 创建可视化图表
plt.figure(figsize=(14, 6))

# 1. 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history['loss'], 'b-', linewidth=2, label='Train Loss')
plt.plot(history['val_loss'], 'r--', linewidth=2, label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 2. AUC曲线
plt.subplot(1, 2, 2)
plt.plot(history['auc'], 'g-', linewidth=2, label='Train AUC')
plt.plot(history['val_auc'], 'm--', linewidth=2, label='Validation AUC')
plt.title('Training and Validation AUC', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 调整布局并保存
plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印最终指标
print(f"Final Training Loss: {history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
print(f"Final Training AUC: {history['auc'][-1]:.4f}")
print(f"Final Validation AUC: {history['val_auc'][-1]:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. 使用模型进行预测（假设X_val_dynamic和X_val_static是验证集数据）
y_pred_proba = model.predict({'dynamic_input': X_dynamic, 'static_input': X_static})
y_pred = (y_pred_proba > 0.5).astype(int)  # 默认阈值0.5

# 2. 计算混淆矩阵
cm = confusion_matrix(y, y_pred)
print("原始混淆矩阵:\n", cm)

# 3. 绘制专业化的混淆矩阵
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-VTE", "VTE"]
)
disp.plot(
    cmap='Blues',
    values_format='d',
    colorbar=False,
    ax=plt.gca()
)

# 4. 添加自定义美化
plt.title("Confusion Matrix (Threshold=0.5)", fontsize=14, pad=20)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=11, rotation=0)
plt.yticks(fontsize=11, rotation=0)

# 5. 在每个格子添加百分比标注
total = cm.sum()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i,
                f"{cm[i, j]}\n({cm[i, j]/total:.1%})",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black",
                fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 打印分类报告
print("\nDetailed Classification Report:")
print(classification_report(y, y_pred, target_names=["Non-VTE", "VTE"]))