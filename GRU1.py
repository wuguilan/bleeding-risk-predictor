import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras import regularizers
import tensorflow as tf
# 1. 读取静态和动态数据
static_data = pd.read_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\MIMIC-static.csv')  # 静态数据
dynamic_data = pd.read_csv('D:\\所有工作进展\\毕业学位论文相关\\数据\\dynamic_data_mean_imputed.csv')  # 动态数据

# 2. 确保静态数据和动态数据按 'stay_id' 对齐
merged_data = pd.merge(dynamic_data, static_data, on='stay_id', how='left')

# 3. 提取静态特征和动态特征
static_features = merged_data[['stay_id', "sofa", "age", "gender", "congestive_heart_failure", "cerebrovascular_disease",
                               "diabetes", "renal_disease", "malignant_cancer", "severe_liver_disease",
                               "metastatic_solid_tumor", "mv", "used_antibiotic_24h", "used_combo_antibiotics_24h",
                               "cephalosporins", "fluoroquinolones", "vancomycin", "cvc", "vasopressors",
                               "apsiii", "history_vte", "history_cancer", "bedrest", "BMI"]]  # 静态特征列

dynamic_features = merged_data[['stay_id', 'time_window', "inr", "pt", "ptt", "hematocrit", "hemoglobin", "platelet", "wbc",
                                "alt", "alp", "ast", "bilirubin_total", "bun", "creatinine", "glucose", "lactate",
                                "heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "gcs"]]  # 动态特征列

labels = merged_data['vte_y'].values  # 目标标签
# 转换为三维数组形状 (4, 3, 1)
data_3d = dynamic_features[:, :, np.newaxis]

print(data_3d)
# 5. 使用 GridSearch 进行参数搜索
def build_model(gru_units=40, learning_rate=0.001, l2_lambda=0.01):
    static_input = Input(shape=(static_features_expanded.shape[1], static_features_expanded.shape[2]))  # 静态特征输入
    dynamic_input = Input(shape=(dynamic_features.shape[1], dynamic_features.shape[1]))  # 动态特征输入

    # 静态特征处理
    static_output = Dense(gru_units, activation='sigmoid',
                          kernel_initializer=GlorotNormal(),
                          kernel_regularizer=regularizers.l2(l2_lambda))(static_input)

    # 动态特征处理
    gru_output = GRU(gru_units, activation='sigmoid',
                     kernel_initializer=GlorotNormal(),
                     recurrent_initializer=GlorotNormal(),
                     return_sequences=False,
                     kernel_regularizer=regularizers.l2(l2_lambda))(dynamic_input)

    # 合并静态特征和动态特征
    merged = Concatenate()([static_output, gru_output])
    final_output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[static_input, dynamic_input], outputs=final_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# 6. 网格搜索和模型训练
param_grid = {
    'gru_units': [32, 40, 48],  # 隐藏单元数候选
    'learning_rate': [0.001, 0.0005],  # 学习率候选
    'l2_lambda': [0.01, 0.001, 0.0001]  # L2 正则化强度候选
}

# 7. 10折交叉验证 + 网格搜索
keras_classifier = KerasClassifier(build_fn=build_model, epochs=5, batch_size=32, verbose=0)
grid_search = GridSearchCV(
    estimator=keras_classifier,
    param_grid=param_grid,
    cv=KFold(n_splits=10, shuffle=True),  # 10折交叉验证
    scoring='accuracy',
    n_jobs=-1
)
print(static_features_expanded.shape)
print(dynamic_features.shape)

# 8. 进行训练
grid_search.fit([static_features_expanded, dynamic_features.values], labels)

# 9. 打印网格搜索结果
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# 10. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
predictions = best_model.predict([static_features_expanded, dynamic_features.values])

# 11. 输出预测结果（例如：前5个预测）
print(predictions[:5])
