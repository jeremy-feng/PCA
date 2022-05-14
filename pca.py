import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
待办：
- 支持多种文件格式
- 展示输入文件示例
- 输出主成分结果的列名和行名应该容易阅读
- 输出矢量图，图片大小合适
"""
    )

uploaded_file = st.file_uploader("上传文件", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, index_col=0)


    # 求出data的协方差矩阵
    V = np.cov(data.T)
    # 对data进行主成分分析
    pca = PCA() # create a PCA object
    # 求出主成分
    R = pca.fit_transform(data)
    # 求出主成分系数，即特征向量
    coeff = pca.components_
    # 求出主成分方差，即各特征向量对应的特征值
    latent = pca.explained_variance_
    # 求出每一个主成分所贡献的比例
    explained = pca.explained_variance_ratio_

    score = np.dot(data, coeff)
    st.dataframe(score)

    # 设置最多绘制的主成分个数
    max_pc_plotted = st.number_input(label='最多绘制的主成分个数', value=3, step=1, min_value=1)

    # 绘制方差比例碎石图
    per_variance_ratio = np.round(explained* 100, decimals=1)
    number_of_pc_plotted = min(len(per_variance_ratio), max_pc_plotted)
    per_variance_ratio = per_variance_ratio[0:number_of_pc_plotted]
    labels = ['第' + str(x) + '主成分' for x in range(1, number_of_pc_plotted+1)]
    plt.bar(x=range(1,number_of_pc_plotted+1), height=per_variance_ratio, tick_label=labels)
    plt.ylabel('方差比例（%）')
    plt.xlabel('主成分')
    plt.title('方差比例碎石图')
    st.pyplot(plt)

