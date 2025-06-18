# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from tensorflow.keras import layers, models, initializers, optimizers, callbacks
import shap
import seaborn as sns
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

# ----------------- 数据加载 -------------------
print("步骤1: 加载数据...")
static_df = pd.read_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\MIMIC-static.csv')
dynamic_df = pd.read_csv("D:\\所有工作进展\\毕业学位论文相关\\数据\\dynamic_data_mean_imputed.csv")

# 外部验证数据
external_static = pd.read_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-static-BMI.csv')
external_dynamic = pd.read_csv("D:\\所有工作进展\\毕业学位论文相关\\数据\\eicu-dyn-merged.csv")

# ----------------- 数据预处理 -------------------
print("\n步骤2: 数据预处理...")

# 动态特征处理
dynamic_df = dynamic_df.sort_values(['stay_id', 'time_window'])
feature_cols = [col for col in dynamic_df.columns if col not in ['stay_id', 'time_window', "vte"]]
X_dynamic = np.stack([group[feature_cols].to_numpy() for _, group in dynamic_df.groupby("stay_id")])

# 静态特征处理
labels = static_df['vte'].values
groups = static_df['stay_id'].values
static_features = static_df.drop(columns=['stay_id', 'vte']).values
static_feature_names = static_df.drop(columns=['stay_id', 'vte']).columns.tolist()

# 检查数据维度
print(f"静态特征维度: {static_features.shape}")
print(f"动态特征维度: {X_dynamic.shape}")
assert dynamic_df.groupby("stay_id").size().eq(4).all(), "错误: 存在时间步不为4的患者"

# ----------------- 数据划分 -------------------
print("\n步骤3: 划分训练测试集...")
np.random.seed(42)
unique_ids = np.unique(groups)
np.random.shuffle(unique_ids)

train_ids = unique_ids[:int(0.9 * len(unique_ids))]
test_ids = unique_ids[int(0.9 * len(unique_ids)):]

train_mask = np.isin(groups, train_ids)
test_mask = np.isin(groups, test_ids)

X_static_train, X_static_test = static_features[train_mask], static_features[test_mask]
X_dynamic_train, X_dynamic_test = X_dynamic[train_mask], X_dynamic[test_mask]
y_train, y_test = labels[train_mask], labels[test_mask]
groups_train = groups[train_mask]

#--以上已理解
# ----------------- 标准化处理 -------------------
print("\n步骤4: 数据标准化...")
scaler_static = StandardScaler().fit(X_static_train) #-先提取训练集的均值和标准差
X_static_train = scaler_static.transform(X_static_train) #-用训练集的均值和标准差训练标准化处理
X_static_test = scaler_static.transform(X_static_test)

scaler_dyn = StandardScaler().fit(X_dynamic_train.reshape(-1, X_dynamic_train.shape[2]))
X_dynamic_train = scaler_dyn.transform(
    X_dynamic_train.reshape(-1, X_dynamic_train.shape[2])).reshape(X_dynamic_train.shape)
X_dynamic_test = scaler_dyn.transform(
    X_dynamic_test.reshape(-1, X_dynamic_test.shape[2])).reshape(X_dynamic_test.shape)

from tensorflow.keras import layers, models, optimizers


def build_model(input_dynamic_shape, input_static_dim):
    # 动态特征输入
    dynamic_input = layers.Input(shape=input_dynamic_shape, name='dynamic_input')

    # 使用GRU层处理动态时间序列
    gru_out = layers.GRU(64, return_sequences=False)(dynamic_input)
    gru_out = layers.BatchNormalization()(gru_out)  # 批归一化
    gru_out = layers.Dropout(0.3)(gru_out)  # Dropout防止过拟合
    # 静态特征输入
    static_input = layers.Input(shape=(input_static_dim,), name='static_input')
    static_dense = layers.Dense(32, activation='relu')(static_input)

    # 特征融合
    merged = layers.Concatenate()([gru_out, static_dense])

    # 输出层
    output = layers.Dense(1, activation='sigmoid')(merged)

    # 构建并编译模型
    model = models.Model(inputs=[dynamic_input, static_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['AUC', 'accuracy'])

    return model

# ----------------- 训练配置 -------------------
# 修改回调函数配置
callbacks = [
    callbacks.EarlyStopping(
        patience=10,
        monitor='val_auc',
        mode='max',
        restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_auc'
    ),
    callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )
]
# ----------------- 交叉验证训练并预测 -------------------
print("\n步骤5: 开始交叉验证训练...")
gkf = GroupKFold(n_splits=5)
test_preds = []
models_list = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_static_train, y_train, groups_train)):
    print(f"\n训练第 {fold + 1} 折...")

    # 数据准备
    X_stat_tr, X_dyn_tr = X_static_train[train_idx], X_dynamic_train[train_idx]
    X_stat_val, X_dyn_val = X_static_train[val_idx], X_dynamic_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # 类别不平衡处理
    flat_X = np.concatenate([
        X_dyn_tr.reshape(len(X_dyn_tr), -1),
        X_stat_tr
    ], axis=1)
    ada = ADASYN(random_state=42)
    flat_X_res, y_res = ada.fit_resample(flat_X, y_tr)
    X_stat_tr_res = flat_X_res[:, -X_stat_tr.shape[1]:]
    X_dyn_tr_res = flat_X_res[:, :-X_stat_tr.shape[1]].reshape(-1, *X_dyn_tr.shape[1:])

    # 模型构建
    model = build_model(input_dynamic_shape=X_dyn_tr.shape[1:], input_static_dim=X_stat_tr.shape[1])

    # 模型训练
    model.fit(
        [X_dyn_tr_res, X_stat_tr_res], y_res,
        validation_data=([X_dyn_val, X_stat_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 保存模型
    models_list.append(model)

    # 在测试集上预测并保存
    test_pred = model.predict([X_dynamic_test, X_static_test]).flatten()
    test_preds.append(test_pred)

# ----------------- 模型评估 -------------------
print("\n步骤6: 模型评估...")
final_pred = np.mean(test_preds, axis=0)

print("Has NaN:", np.isnan(final_pred).any())
print("Has Inf:", np.isinf(final_pred).any())
print("预测类型:", type(test_preds))
print("预测折数:", len(test_preds))
print("每折预测 shape:", [arr.shape for arr in test_preds])

# ROC AUC
test_auc = roc_auc_score(y_test, final_pred)
print(f"\n测试集 AUC: {test_auc:.4f}")

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, final_pred > 0.5, target_names=['非VTE', 'VTE']))
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, final_pred > 0.5)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['非VTE', 'VTE'], yticklabels=['非VTE', 'VTE'])


# ----------------- 可视化分析 -------------------
def plot_attention_chinese(model, sample_data, feature_names):
    """可视化注意力权重(中文版)"""
    print("\n生成注意力权重可视化...")

    # 获取注意力层输出
    attention_model = models.Model(
        inputs=model.input[0],
        outputs=model.get_layer('dynamic_attention').output
    )

    # 随机选择样本
    sample_idx = np.random.choice(len(sample_data), 5, replace=False)
    sample = sample_data[sample_idx]

    # 计算注意力权重
    attention_weights = attention_model.predict(sample)
    weights_avg = np.mean(attention_weights, axis=0)

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(weights_avg.T,
                xticklabels=[f'时间窗{i + 1}' for i in range(4)],
                yticklabels=feature_names,
                cmap='YlOrRd')
    plt.title('动态特征注意力权重分布')
    plt.xlabel('时间窗口')
    plt.ylabel('临床特征')
    plt.tight_layout()
    plt.show()


# 调用可视化函数
plot_attention_chinese(models_list[0], X_dynamic_train, feature_cols)

# ----------------- 外部验证 -------------------
print("\n步骤7: 外部验证...")
# 外部数据预处理（修正版）

# 1. 静态特征处理
external_static_features = external_static.drop(columns=['stay_id', 'vte']).values
external_labels = external_static['vte'].values
external_static_scaled = scaler_static.transform(external_static_features)

# 2. 动态特征处理（关键修正部分）
# 确保外部动态数据与训练数据相同的预处理流程
external_dynamic_sorted = external_dynamic.sort_values(['stay_id', 'time_window'])
external_dynamic_array = np.stack([
    group[feature_cols].to_numpy()
    for _, group in external_dynamic_sorted.groupby("stay_id")
])

# 检查维度一致性
print(f"外部动态数据原始维度: {external_dynamic_array.shape}")
print(f"训练数据维度参考: {X_dynamic_train.shape}")

# 标准化处理
external_dynamic_scaled = scaler_dyn.transform(
    external_dynamic_array.reshape(-1, external_dynamic_array.shape[2])
).reshape(external_dynamic_array.shape)

# 3. 预测并评估
try:
    external_pred = np.mean([
        model.predict([external_dynamic_scaled, external_static_scaled])
        for model in models_list
    ], axis=0)
    external_auc = roc_auc_score(external_labels, external_pred)
    print(f"外部验证AUC: {external_auc:.4f}")

    # 外部验证分类报告
    print("\n外部验证分类报告:")
    print(classification_report(
        external_labels,
        external_pred > 0.5,
        target_names=['非VTE', 'VTE']
    ))
except Exception as e:
    print(f"外部验证出错: {str(e)}")
    # 添加详细错误诊断
    print(f"输入数据形状 - 动态: {external_dynamic_scaled.shape}")
    print(f"输入数据形状 - 静态: {external_static_scaled.shape}")
# ----------------- 可解释性分析 -------------------
def chinese_shap_analysis(model, sample_data, feature_names):
    """内存友好的SHAP分析(中文版)"""
    print("\n进行SHAP特征重要性分析...")

    # 使用小样本分析
    background = sample_data[:50]
    explainer = shap.DeepExplainer(model, background)

    # 计算SHAP值
    test_samples = sample_data[50:150]
    shap_values = explainer.shap_values(test_samples)

    # 可视化
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, test_samples,
                      feature_names=feature_names,
                      plot_type="bar",
                      show=False)
    plt.title('特征重要性分析')
    plt.tight_layout()
    plt.show()


# 使用第一个模型进行分析
sample_data = [X_dynamic_train[:100], X_static_train[:100]]
chinese_shap_analysis(models_list[0], sample_data, feature_cols + static_feature_names)

print("\n所有分析完成!")