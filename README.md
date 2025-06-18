## 功能
- 输入患者临床参数（APACHE评分、血小板计数等）
- 输出出血风险概率及SHAP解释图

## 快速开始
```bash
# 克隆仓库
git clone https://github.com/wuguilan/bleeding-risk-predictor.git
cd bleeding-risk-predictor

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run bleeding-risk-predictor.py
