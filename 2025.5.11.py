import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import gc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, auc, average_precision_score
)
from sklearn.calibration import calibration_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import ADASYN
from tqdm import tqdm
import tensorflow as tf
import joblib

# === 配置参数 ===
DATA_PATH = 'D:\\所有工作进展\\毕业学位论文相关\\数据\\merged_static_dynamic_data.csv'
MODEL_SAVE_DIR = 'saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    filename=os.path.join(MODEL_SAVE_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'patient_id' not in df.columns:
        df['patient_id'] = df['stay_id']
    return df


def preprocess_data(df, dynamic_cols):
    for col in dynamic_cols:
        if col in df.columns:
            df[col] = df.groupby('stay_id')[col].ffill().bfill()
            df[col].fillna(df[col].median(), inplace=True)
    return df


def normalize_data(train_df, val_df, feature_cols):
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.pkl'))
    return train_df, val_df


def make_sequences(df, feature_cols, max_len=None):
    X, y, ids, static_features = [], [], [], []
    for pid, group in df.groupby('stay_id'):
        X.append(group[feature_cols].values)
        y.append(group['vte'].iloc[0])
        ids.append(pid)
        static_features.append(group.iloc[0][static_columns].values)
    X_padded = pad_sequences(X, padding='post', dtype='float32', maxlen=max_len)
    static_array = np.array(static_features).astype(np.float32)
    return X_padded, static_array, np.array(y), np.array(ids)


def build_gru_model(input_dim, static_dim, units=40, learning_rate=0.001):
    dyn_input = Input(shape=(None, input_dim), name='dynamic_input')
    static_input = Input(shape=(static_dim,), name='static_input')
    x = GRU(units)(dyn_input)
    concat = Concatenate()([x, static_input])
    output = Dense(1, activation='sigmoid')(concat)
    model = Model(inputs=[dyn_input, static_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


def evaluate_predictions(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_pred_prob),
        'F1': f1_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred),
        'Specificity': tn / (tn + fp),
        'PPV': precision_score(y_true, y_pred),
        'NPV': tn / (tn + fn),
        'AP': average_precision_score(y_true, y_pred_prob)
    }


def train_with_groupkfold(df, static_cols, dynamic_cols, n_splits=5):
    df = preprocess_data(df, dynamic_cols)
    feature_cols = static_cols + dynamic_cols
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['vte'], df['stay_id'])):
        print(f"=== Fold {fold + 1}/{n_splits} ===")
        train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
        train_df, val_df = normalize_data(train_df, val_df, feature_cols)

        X_train_dyn, X_train_static, y_train, _ = make_sequences(train_df, dynamic_cols)
        X_val_dyn, X_val_static, y_val, _ = make_sequences(val_df, dynamic_cols)

        # 过采样
        X_dyn_flat = X_train_dyn[:, -1, :]  # 使用最后一个时间步
        adasyn = ADASYN()
        X_resampled, y_resampled = adasyn.fit_resample(np.hstack([X_dyn_flat, X_train_static]), y_train)
        X_dyn_resampled = X_resampled[:, :X_dyn_flat.shape[1]]
        X_static_resampled = X_resampled[:, X_dyn_flat.shape[1]:]
        X_dyn_resampled = np.tile(X_dyn_resampled[:, np.newaxis, :], (1, X_train_dyn.shape[1], 1))

        model = build_gru_model(input_dim=X_train_dyn.shape[2], static_dim=X_train_static.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit([X_dyn_resampled, X_static_resampled], y_resampled,
                  validation_data=([X_val_dyn, X_val_static], y_val),
                  epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1)

        y_val_prob = model.predict([X_val_dyn, X_val_static], batch_size=32)
        metrics = evaluate_predictions(y_val, y_val_prob)
        print("\n=== 模型评估结果 ===")
        print(f"{'指标':<15}{'值':<10}{'解释':<30}")
        print("-" * 55)
        print(f"{'AUC':<15}{metrics['AUC']:.4f}{'0.9以上优秀，0.8-0.9良好'}")
        print(f"{'F1':<15}{metrics['F1']:.4f}{'综合精确率和召回率'}")
        print(f"{'Accuracy':<15}{metrics['Accuracy']:.4f}{'整体准确率'}")
        print(f"{'Sensitivity':<15}{metrics['Sensitivity']:.4f}{'真阳性率（避免漏诊）'}")
        print(f"{'Specificity':<15}{metrics['Specificity']:.4f}{'真阴性率（避免误诊）'}")
        print(f"{'PPV':<15}{metrics['PPV']:.4f}{'阳性预测值'}")
        print(f"{'NPV':<15}{metrics['NPV']:.4f}{'阴性预测值'}")
        print(f"{'AP':<15}{metrics['AP']:.4f}{'平均精确率（PR曲线下面积）'}")
        tn, fp, fn, tp = confusion_matrix(y_val, (y_val_prob > 0.5).astype(int)).ravel()
        print("\n=== 混淆矩阵 ===")
        print(f"              预测阴性   预测阳性")
        print(f"实际阴性  TN={tn:<6} FP={fp:<6}")
        print(f"实际阳性  FN={fn:<6} TP={tp:<6}")


if __name__ == '__main__':
    df = load_data(DATA_PATH)
    static_columns = ['sofa', 'age', 'gender', 'height', 'weight', 'congestive_heart_failure',
                      'cerebrovascular_disease', 'diabetes', 'paraplegia', 'renal_disease',
                      'malignant_cancer', 'severe_liver_disease', 'metastatic_solid_tumor',
                      'charlson_comorbidity_index', 'MV', 'used_antibiotic_24h',
                      'used_combo_antibiotics_24h', 'cephalosporins', 'fluoroquinolones',
                      'vancomycin', 'cvc', 'vasopressors', 'apsiii', 'history_vte',
                      'history_cancer', 'bedrest']
    dynamic_columns = ['inr', 'pt', 'ptt', 'hematocrit', 'hemoglobin', 'platelet', 'wbc', 'alt',
                       'alp', 'ast', 'bilirubin_total', 'bun', 'creatinine', 'glucose', 'lactate',
                       'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'gcs']
    train_with_groupkfold(df, static_columns, dynamic_columns)
