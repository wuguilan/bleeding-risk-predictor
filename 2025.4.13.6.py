import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import gc
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, auc, average_precision_score
)
from sklearn.calibration import calibration_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import tensorflow as tf
import joblib

# === 配置参数 ===
DATA_PATH = 'D:\\所有工作进展\\毕业学位论文相关\\数据\\merged_static_dynamic_data.csv'  # 替换为实际路径
MODEL_SAVE_DIR = 'saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    filename=os.path.join(MODEL_SAVE_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# === 1. 数据加载 ===
def load_data(file_path):
    """加载并验证数据"""
    try:
        df = pd.read_csv(file_path)

        # 关键列检查
        required_cols = ['stay_id', 'vte', 'time_window']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺失必要列: {missing_cols}")

        # 添加patient_id（如果不存在）
        if 'patient_id' not in df.columns:
            df['patient_id'] = df['stay_id']

        logging.info(f"数据加载成功，样本量: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"数据加载失败: {str(e)}")
        raise


# === 2. 数据预处理 ===
def preprocess_data(df, dynamic_cols):
    """处理缺失值和添加时序特征"""
    try:
        # 动态特征填充
        for col in dynamic_cols:
            if col in df.columns:
                # 先按患者分组填充，再全局填充
                df[col] = df.groupby('stay_id')[col].ffill().bfill()
                df[col].fillna(df[col].median(), inplace=True)

        # 添加时序特征
        for col in dynamic_cols:
            if col in df.columns:
                df[f'{col}_diff'] = df.groupby('stay_id')[col].diff().fillna(0)
                df[f'{col}_rolling_mean'] = df.groupby('stay_id')[col].rolling(3, min_periods=1).mean().reset_index(
                    level=0, drop=True)
                df[f'{col}_rolling_std'] = df.groupby('stay_id')[col].rolling(3, min_periods=1).std().reset_index(
                    level=0, drop=True)

        logging.info("数据预处理完成")
        return df
    except Exception as e:
        logging.error(f"数据预处理失败: {str(e)}")
        raise


# === 3. 数据标准化 ===
def normalize_data(train_df, val_df, feature_cols):
    """标准化处理（仅用训练集统计量）"""
    try:
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])

        # 保存标准化器
        scaler_path = os.path.join(MODEL_SAVE_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)

        return train_df, val_df, scaler
    except Exception as e:
        logging.error(f"数据标准化失败: {str(e)}")
        raise


# === 4. 序列生成 ===
def make_sequences(df, feature_cols, max_len=None):
    """生成等长时序序列"""
    try:
        X, y, ids = [], [], []
        for pid, group in df.groupby('stay_id'):
            X.append(group[feature_cols].values)
            y.append(group['vte'].iloc[0])  # 同一患者所有时间片相同标签
            ids.append(pid)

        # 自动填充到最大长度
        X_padded = pad_sequences(X, padding='post', dtype='float32', maxlen=max_len)
        logging.info(f"生成序列完成，样本量: {len(X_padded)}")
        return X_padded, np.array(y), np.array(ids)
    except Exception as e:
        logging.error(f"序列生成失败: {str(e)}")
        raise


# === 5. 模型构建 ===
def build_gru_model(input_dim, units=40, learning_rate=0.001):
    """构建GRU模型"""
    model = Sequential([
        GRU(units, input_shape=(None, input_dim)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# === 6. 评估指标 ===
def evaluate_predictions(y_true, y_pred_prob, threshold=0.1):
    """综合评估指标"""
    try:
        y_pred = (y_pred_prob > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            'AUC': roc_auc_score(y_true, y_pred_prob),
            'F1': f1_score(y_true, y_pred),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Sensitivity': recall_score(y_true, y_pred),
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'PPV': precision_score(y_true, y_pred),
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'AP': average_precision_score(y_true, y_pred_prob)
        }
    except Exception as e:
        logging.error(f"评估失败: {str(e)}")
        raise


# === 7. 可视化工具 ===
def plot_and_save(y_true, y_prob, plot_type='roc', save_dir=MODEL_SAVE_DIR):
    """统一可视化函数"""
    try:
        plt.figure(figsize=(8, 6))

        if plot_type == 'roc':
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend()
            plt.show()

        elif plot_type == 'calibration':
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Actual probability')
            plt.title('Calibration Curve')
            plt.legend()
            plt.show()

        elif plot_type == 'dca':
            thresholds = np.linspace(0.01, 0.99, 99)
            net_benefits = []
            prevalence = np.mean(y_true)

            for pt in thresholds:
                pred = (y_prob >= pt).astype(int)
                tp = np.sum((y_true == 1) & (pred == 1))
                fp = np.sum((y_true == 0) & (pred == 1))
                net_benefit = (tp / len(y_true)) - (fp / len(y_true)) * (pt / (1 - pt))
                net_benefits.append(net_benefit)

            plt.plot(thresholds, net_benefits, label='Model')
            plt.plot(thresholds, [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds],
                     '--', label='Treat All')
            plt.plot(thresholds, [0] * len(thresholds), ':', label='Treat None')
            plt.xlabel('Threshold Probability')
            plt.ylabel('Net Benefit')
            plt.title('Decision Curve Analysis')
            plt.xlim([0, 0.1])
            plt.ylim([-0.1, max(net_benefits) * 1.1])
            plt.legend()
            plt.show()

        plt.legend(loc='best')
        save_path = os.path.join(save_dir, f'{plot_type}_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"已保存 {plot_type} 曲线到 {save_path}")
    except Exception as e:
        logging.error(f"可视化失败: {str(e)}")
        raise


# === 8. 主训练流程 ===
def train_and_evaluate(df, static_cols, dynamic_cols, epochs=50, batch_size=32):
    """训练和评估模型"""
    try:
        # 数据预处理
        df = preprocess_data(df, dynamic_cols)

        # 划分数据集
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # 数据标准化
        feature_cols = static_cols + dynamic_cols
        train_df, val_df, scaler = normalize_data(train_df, val_df, feature_cols)

        # 序列生成
        X_train, y_train, _ = make_sequences(train_df, feature_cols)
        X_val, y_val, _ = make_sequences(val_df, feature_cols)

        # 模型构建
        input_dim = X_train.shape[2]
        model = build_gru_model(input_dim)

        # 模型训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                  callbacks=[early_stopping], verbose=1)

        # 模型评估
        y_val_prob = model.predict(X_val, batch_size=batch_size)
        metrics = evaluate_predictions(y_val, y_val_prob)
        # === 新增：控制台输出评估结果 ===
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

        # 打印混淆矩阵
        y_pred = (y_val_prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        print("\n=== 混淆矩阵 ===")
        print(f"              预测阴性   预测阳性")
        print(f"实际阴性  TN={tn:<6} FP={fp:<6}")
        print(f"实际阳性  FN={fn:<6} TP={tp:<6}")

        # 打印评估结果
        logging.info("评估结果:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        # 可视化结果
        plot_and_save(y_val, y_val_prob, plot_type='roc')
        plot_and_save(y_val, y_val_prob, plot_type='calibration')
        plot_and_save(y_val, y_val_prob, plot_type='dca')

        # 保存模型
        model_save_path = os.path.join(MODEL_SAVE_DIR, 'gru_model.h5')
        model.save(model_save_path)
        logging.info(f"模型保存至: {model_save_path}")

    except Exception as e:
        logging.error(f"训练失败: {str(e)}")
        raise


# === 9. 执行训练 ===
if __name__ == '__main__':
    # 加载数据
    df = load_data(DATA_PATH)

    # 设置静态和动态特征列
    static_columns = ['sofa', 'age', 'gender', 'height', 'weight', 'congestive_heart_failure',
        'cerebrovascular_disease', 'diabetes', 'paraplegia', 'renal_disease',
        'malignant_cancer', 'severe_liver_disease', 'metastatic_solid_tumor',
        'charlson_comorbidity_index', 'MV', 'used_antibiotic_24h',
        'used_combo_antibiotics_24h', 'cephalosporins', 'fluoroquinolones',
        'vancomycin', 'cvc', 'vasopressors', 'apsiii', 'history_vte',
        'history_cancer', 'bedrest']  # 请替换为实际的静态特征列
    dynamic_columns = [ 'inr', 'pt', 'ptt', 'hematocrit', 'hemoglobin', 'platelet', 'wbc', 'alt',
        'alp', 'ast', 'bilirubin_total', 'bun', 'creatinine', 'glucose', 'lactate',
        'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'gcs']  # 请替换为实际的动态特征列

    # 开始训练
    train_and_evaluate(df, static_columns, dynamic_columns)
