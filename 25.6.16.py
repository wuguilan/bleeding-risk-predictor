import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# 假设你已经加载了数据
df = pd.read_csv("D:\\所有工作进展\\投稿相关\可解释的机器学习模型（Alfalfa-ICU-MB）预测重症病房住院期间的大出血风险：一项基于208个中心数据的分析\\数据\\统计数据.csv")

# 特征列表
features = [
    'apache_iv_score', 'gcs', 'albumin_max', 'hematocrit_min', 'anemia',
    'platelet_min', 'ptt_max', 'coagulation_dysfunction', 'pt_max', 'bun_max',
    'respiratoryrate', 'nibp_systolic', 'nibp_diastolic', 'gender', 'caucasian',
    'medsurg_icu', 'cardiac_icu', 'neuro_icu', 'gastrointestinal_condition',
    'trauma', 'history_of_bleed', 'history_of_vte', 'sepsis', 'vascular_disorders',
    'acute_coronary_syndrome', 'respiratory_failure', 'vasopressors_inotropic_agents',
    'stress_ulcer_drug'
]

# 分离特征和标签
X = df[features]
y = df['major_bleed']

# === 2. 划分训练集与测试集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 3. 处理类别不平衡 ===
# 计算正负样本比例
pos = sum(y_train == 1)
neg = sum(y_train == 0)
scale_pos_weight = neg / pos

print(f"正样本: {pos}, 负样本: {neg}, scale_pos_weight: {scale_pos_weight:.2f}")

# === 4. 定义并训练模型 ===
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# === 5. 模型评估 ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("✅ ROC AUC:", roc_auc_score(y_test, y_prob))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# === 6. 保存模型与特征名 ===
joblib.dump({
    "model": model,
    "feature_names": features
}, "xgboost_bleed_model.joblib")

print("✅ 模型与特征名已保存到 'xgboost_bleed_model.joblib'")
