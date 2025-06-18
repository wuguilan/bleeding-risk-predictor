import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# -------------------- 1. 数据加载函数 --------------------
def load_and_preprocess(static_path, dynamic_path):
    """加载并预处理静态和动态特征数据"""
    # 读取数据
    static_df = pd.read_csv(static_path)
    dynamic_df = pd.read_csv(dynamic_path)

    # 数据校验
    assert 'vte' in static_df.columns, "静态数据必须包含vte标签"
    assert 'stay_id' in static_df.columns, "需要stay_id列"

    # 类型转换
    static_df['stay_id'] = static_df['stay_id'].astype(int)
    dynamic_df['stay_id'] = dynamic_df['stay_id'].astype(int)

    # 动态数据时序排序
    if 'time_window' in dynamic_df.columns:
        dynamic_df = dynamic_df.sort_values(['stay_id', 'time_window'])
    else:
        # 如果没有time_step列，假设数据已经是正确顺序
        dynamic_df = dynamic_df.sort_values('stay_id')

    return static_df, dynamic_df


# -------------------- 2. 模型定义 --------------------
class MedicalGRUAttention(nn.Module):
    def __init__(self, input_dim, static_dim, **kwargs):
        super().__init__()
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.dropout = kwargs.get('dropout', 0.3)
        bidirectional = kwargs.get('bidirectional', True)

        # 动态特征处理
        self.gru = nn.GRU(
            input_dim,
            self.hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        gru_out_dim = self.hidden_dim * 2 if bidirectional else self.hidden_dim

        # 注意力机制
        self.attn = nn.Sequential(
            nn.Linear(gru_out_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1))

        # 静态特征处理
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))

        # 联合预测
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1))

    def forward(self, x_dyn, x_stat):
        # 动态特征处理
        gru_out, _ = self.gru(x_dyn)  # [batch, seq_len, hidden*(2 if bidirectional)]

        # 注意力机制
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)

        # 静态特征处理
        static_feat = self.static_net(x_stat)

        # 特征融合与预测
        combined = torch.cat([context, static_feat], dim=1)
        output = torch.sigmoid(self.classifier(combined)).squeeze()

        return output, attn_weights.squeeze()


# -------------------- 3. 训练评估函数 --------------------
def train_and_evaluate(params, X_dyn_train, X_stat_train, y_train, X_dyn_val, X_stat_val, y_val, device):
    """训练并评估模型"""
    # 数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_dyn_train),
        torch.FloatTensor(X_stat_train),
        torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    val_dataset = TensorDataset(
        torch.FloatTensor(X_dyn_val),
        torch.FloatTensor(X_stat_val),
        torch.FloatTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # 模型初始化
    model = MedicalGRUAttention(
        input_dim=X_dyn_train.shape[2],
        static_dim=X_stat_train.shape[1],
        **params
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.BCELoss()

    # 训练循环
    best_auc = 0
    for epoch in range(params['epochs']):
        model.train()
        for x_dyn, x_stat, y in train_loader:
            x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)

            optimizer.zero_grad()
            preds, _ = model(x_dyn, x_stat)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

    # 验证评估
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_dyn, x_stat, y in val_loader:
            x_dyn, x_stat = x_dyn.to(device), x_stat.to(device)
            preds, _ = model(x_dyn, x_stat)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return auc, ap


# -------------------- 4. 主网格搜索函数 --------------------
def grid_search_main():
    """执行网格搜索寻找最优参数"""
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 参数网格
    param_grid = {
        'hidden_dim': [32, 64],  # GRU隐藏层维度
        'lr': [1e-3, 5e-4],  # 学习率
        'batch_size': [32, 64],  # 批大小
        'dropout': [0.2, 0.3],  # Dropout率
        'epochs': [15, 20],  # 训练轮数
        'bidirectional': [True, False]  # 是否使用双向GRU
    }

    # 加载数据（替换为你的实际路径）
    static_path = "D:/所有工作进展/毕业学位论文相关/数据/static_data.csv"
    dynamic_path = "D:/所有工作进展/毕业学位论文相关/数据/dynamic_data_mean_imputed.csv"
    static_df, dynamic_df = load_and_preprocess(static_path, dynamic_path)

    # 特征工程（根据你的实际特征调整）
    static_features = ["sofa", "age", "gender", "congestive_heart_failure", "cerebrovascular_disease", "diabetes",
                       "paraplegia", "renal_disease", "malignant_cancer", "severe_liver_disease",
                       "metastatic_solid_tumor", "charlson_comorbidity_index", "MV", "used_antibiotic_24h",
                       "used_combo_antibiotics_24h", "cephalosporins", "fluoroquinolones", "vancomycin", "cvc",
                       "vasopressors", "apsiii", "history_vte", "history_cancer", "bedrest", "BMI"]  # 示例特征
    X_static = static_df.set_index('stay_id')[static_features]
    y_all = static_df.set_index('stay_id')['vte']


    # 动态特征处理
    dynamic_features = [col for col in dynamic_df.columns if col not in ['stay_id', 'vte','time_window']]
    X_dynamic = dynamic_df.groupby('stay_id')[dynamic_features].apply(lambda x: x.values.tolist()).reset_index(drop=True)

    X_dynamic = np.array(X_dynamic.tolist())

    # 标准化
    static_scaler = StandardScaler()
    dynamic_scaler = MinMaxScaler(feature_range=(-1, 1))

    # 交叉验证
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 减少折数加速搜索

    # 网格搜索
    best_params = None
    best_score = -1
    search_results = []

    for params in tqdm(ParameterGrid(param_grid), desc='网格搜索进度'):
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_static, y_all)):
            # 数据划分
            train_ids = X_static.index[train_idx]
            val_ids = X_static.index[val_idx]

            # 静态特征处理
            X_stat_train = static_scaler.fit_transform(X_static.loc[train_ids])
            X_stat_val = static_scaler.transform(X_static.loc[val_ids])

            # 动态特征处理
            train_mask = [i for i, sid in enumerate(static_df['stay_id']) if sid in train_ids]
            val_mask = [i for i, sid in enumerate(static_df['stay_id']) if sid in val_ids]

            X_dyn_train = dynamic_scaler.fit_transform(
                X_dynamic[train_mask].reshape(-1, X_dynamic.shape[2])
            ).reshape(-1, *X_dynamic.shape[1:])
            X_dyn_val = dynamic_scaler.transform(
                X_dynamic[val_mask].reshape(-1, X_dynamic.shape[2])
            ).reshape(-1, *X_dynamic.shape[1:])

            # SMOTE处理
            smote = SMOTE(random_state=42)
            X_stat_train_res, y_train_res = smote.fit_resample(X_stat_train, y_all.loc[train_ids])

            # 重新对齐动态特征
            resampled_ids = train_ids[y_train_res.index % len(train_ids)]
            dyn_train_res_mask = [i for i, sid in enumerate(static_df['stay_id']) if sid in resampled_ids]
            X_dyn_train_res = X_dynamic[dyn_train_res_mask]

            # 训练评估
            auc, ap = train_and_evaluate(
                params,
                X_dyn_train_res, X_stat_train_res, y_train_res,
                X_dyn_val, X_stat_val, y_all.loc[val_ids],
                device
            )
            fold_scores.append(auc)

        # 计算平均性能
        mean_score = np.mean(fold_scores)
        params_copy = params.copy()
        params_copy['mean_auc'] = mean_score
        search_results.append(params_copy)

        # 更新最佳参数
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # 结果分析
    results_df = pd.DataFrame(search_results)
    results_df = results_df.sort_values('mean_auc', ascending=False)

    print("\n=== 网格搜索最佳参数 ===")
    print(results_df.head())

    print("\n=== 最优参数组合 ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print(f"验证集平均AUC: {best_score:.4f}")

    return best_params


if __name__ == "__main__":
    optimal_params = grid_search_main()
    print("\n网格搜索完成！最优参数已返回。")