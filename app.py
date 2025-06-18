
import streamlit as st
import time

# 使用 @st.cache_data 装饰器缓存函数的输出
@st.cache_data
def expensive_computation(a, b):
    time.sleep(5)  # 模拟一个耗时的计算过程
    return a * b

a = st.slider('输入a', 0, 10, 5)
b = st.slider('输入b', 0, 10, 5)
c = st.slider('输入b', 0, 10, 5)
d = st.slider('输入b', 0, 10, 5)

# 调用被缓存的函数
result = expensive_computation(a, b)

st.write(f'计算结果：{result}')