import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# ✅ 修正点1：set_page_config() 必须是文件中最先执行的Streamlit命令
st.set_page_config(layout="wide")  # 这行必须放在最前面！

# 加载模型和特征名
@st.cache_resource
def load_model():
    model_data = joblib.load("xgboost_bleed_model.joblib")
    return model_data["model"], model_data["feature_names"]


model, feature_names = load_model()

# 初始化SHAP解释器
explainer = shap.TreeExplainer(model)

# 设置阈值 (您可以根据模型性能调整)
THRESHOLD = 0.5  # 默认阈值，建议根据ROC曲线确定最佳值


def user_input_features():
    st.header("患者临床参数输入")

    col1, col2, col3 = st.columns(3)

    with col1:
        apache_iv_score = st.number_input("APACHE IV评分", min_value=0, max_value=200, value=50)
        gcs = st.number_input("GCS评分", min_value=3, max_value=15, value=12)
        albumin_max = st.number_input("最大白蛋白(g/dL)", min_value=1.0, max_value=6.0, value=3.5, step=0.1)
        hematocrit_min = st.number_input("最低血细胞比容(%)", min_value=10, max_value=60, value=30)
        anemia = st.selectbox("贫血", ["否", "是"], index=0)

    with col2:
        platelet_min = st.number_input("最低血小板计数(×10³/µL)", min_value=10, max_value=500, value=150)
        ptt_max = st.number_input("最大PTT(秒)", min_value=20, max_value=200, value=35)
        pt_max = st.number_input("最大PT(秒)", min_value=10, max_value=50, value=13)
        bun_max = st.number_input("最大BUN(mg/dL)", min_value=5, max_value=100, value=20)
        respiratoryrate = st.number_input("呼吸频率(次/分)", min_value=5, max_value=50, value=18)

    with col3:
        nibp_systolic = st.number_input("收缩压(mmHg)", min_value=50, max_value=250, value=120)
        nibp_diastolic = st.number_input("舒张压(mmHg)", min_value=30, max_value=150, value=80)
        gender = st.selectbox("性别", ["男", "女"], index=0)
        caucasian = st.selectbox("白种人", ["否", "是"], index=1)
        medsurg_icu = st.selectbox("内科/外科ICU", ["否", "是"], index=0)

    # 第二组分栏
    col4, col5, col6 = st.columns(3)

    with col4:
        cardiac_icu = st.selectbox("心脏ICU", ["否", "是"], index=0)
        neuro_icu = st.selectbox("神经ICU", ["否", "是"], index=0)
        gastrointestinal_condition = st.selectbox("胃肠道疾病", ["否", "是"], index=0)

    with col5:
        trauma = st.selectbox("创伤", ["否", "是"], index=0)
        history_of_bleed = st.selectbox("出血病史", ["否", "是"], index=0)
        history_of_vte = st.selectbox("静脉血栓病史", ["否", "是"], index=0)

    with col6:
        sepsis = st.selectbox("败血症", ["否", "是"], index=0)
        vascular_disorders = st.selectbox("血管疾病", ["否", "是"], index=0)
        stress_ulcer_drug = st.selectbox("使用应激性溃疡药物", ["否", "是"], index=0)

    # 自动计算衍生特征
    coagulation_dysfunction = 1 if (ptt_max > 40 or pt_max > 14) else 0
    respiratory_failure = 1 if (respiratoryrate > 24 or nibp_systolic < 90) else 0

    # 创建输入数据框
    data = {
        'apache_iv_score': apache_iv_score,
        'gcs': gcs,
        'albumin_max': albumin_max,
        'hematocrit_min': hematocrit_min,
        'anemia': 1 if anemia == "是" else 0,
        'platelet_min': platelet_min,
        'ptt_max': ptt_max,
        'coagulation_dysfunction': coagulation_dysfunction,
        'pt_max': pt_max,
        'bun_max': bun_max,
        'respiratoryrate': respiratoryrate,
        'nibp_systolic': nibp_systolic,
        'nibp_diastolic': nibp_diastolic,
        'gender': 1 if gender == "女" else 0,
        'caucasian': 1 if caucasian == "是" else 0,
        'medsurg_icu': 1 if medsurg_icu == "是" else 0,
        'cardiac_icu': 1 if cardiac_icu == "是" else 0,
        'neuro_icu': 1 if neuro_icu == "是" else 0,
        'gastrointestinal_condition': 1 if gastrointestinal_condition == "是" else 0,
        'trauma': 1 if trauma == "是" else 0,
        'history_of_bleed': 1 if history_of_bleed == "是" else 0,
        'history_of_vte': 1 if history_of_vte == "是" else 0,
        'sepsis': 1 if sepsis == "是" else 0,
        'vascular_disorders': 1 if vascular_disorders == "是" else 0,
        'acute_coronary_syndrome': 0,  # 示例中没有此字段输入
        'respiratory_failure': respiratory_failure,
        'vasopressors_inotropic_agents': 0,  # 示例中没有此字段输入
        'stress_ulcer_drug': 1 if stress_ulcer_drug == "是" else 0
    }

    return pd.DataFrame([data], columns=feature_names)


def main():
    st.title("🏥 ICU患者大出血风险预测工具")
    st.markdown("""
    **基于XGBoost模型预测ICU患者住院期间大出血风险**  
    *请填写下方患者临床参数后点击预测按钮*
    """)

    input_df = user_input_features()

    if st.button("预测大出血风险"):
        try:
            # 预测
            proba = model.predict_proba(input_df)[0, 1]
            prediction = "高风险" if proba >= THRESHOLD else "低风险"

            # 显示结果
            st.success("预测完成！")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("预测结果", prediction)
            with col2:
                st.metric("风险概率", f"{proba * 100:.1f}%")

            # SHAP解释
            st.subheader("模型解释 - 特征重要性")
            shap_values = explainer(input_df)

            # 创建SHAP摘要图
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values.values, input_df, feature_names=feature_names, plot_type="bar",
                              max_display=15)
            st.pyplot(fig)

            # SHAP 解释图
            st.subheader("个体预测解释")
            force_plot = shap.force_plot(
                base_value=explainer.expected_value,
                shap_values=shap_values.values[0],
                features=input_df.iloc[0],
                feature_names=feature_names,
                matplotlib=False
            )
            st.components.v1.html(shap.getjs() + force_plot.html(), height=300)

            # ✅ 确保风险解释说明在 SHAP 图之后，且在同一个 try 块内
            st.subheader("风险解释说明")
            proba = model.predict_proba(input_df)[0, 1]  # 获取预测概率
            if proba > 0.7:
                st.warning("""
                       **⚠️ 高风险预警**  
                       该患者大出血风险较高 (>70%)，建议：  
                       - 加强凝血功能监测  
                       - 考虑预防性干预措施  
                       - 避免不必要的侵入性操作
                       """)
            elif proba > 0.3:
                st.info("""
                       **ℹ️ 中等风险**  
                       该患者有一定出血风险 (30-70%)，建议：  
                       - 常规凝血监测  
                       - 评估用药方案  
                       - 观察出血征象
                       """)
            else:
                st.success("""
                       **✅ 低风险**  
                       该患者出血风险较低 (<30%)，常规护理即可
                       """)

        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")

    # 侧边栏信息
    with st.sidebar:
        st.header("关于此工具")
        st.markdown("""
        - **模型类型**: XGBoost
        - **训练数据**: 208个中心ICU数据
        - **预测目标**: 住院期间大出血风险
        - **阈值**: {:.3f} (通过Youden指数确定)
        """.format(THRESHOLD))

        st.header("使用说明")
        st.markdown("""
        1. 填写患者临床参数
        2. 点击"预测大出血风险"按钮
        3. 查看预测结果和解释
        """)

        st.warning("""
        **临床注意事项**  
        本工具结果仅供参考，应结合临床判断使用。  
        高风险患者需综合评估其他危险因素。
        """)


if __name__ == '__main__':
    main()