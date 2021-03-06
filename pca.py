import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
已完成：
- 支持多种文件格式
- 输出主成分结果的列名和行名应该容易阅读
待办：
- 展示输入文件示例
- 输出矢量图，图片大小合适
"""
            )

uploaded_file = st.file_uploader("上传文件（支持csv和xlsx格式）", type=["csv", "xlsx"])
if uploaded_file is not None:
    file_name = uploaded_file.name
    # 求文件的后缀名
    file_ext = file_name.split('.')[-1]
    if file_ext == 'csv':
        data = pd.read_csv(uploaded_file, index_col=0)
    elif file_ext == 'xlsx':
        data = pd.read_excel(uploaded_file, index_col=0)
    else:
        st.error("不支持的文件格式！")
        data = None
    st.write("上传的数据：")
    st.write(data)

    # 求出data的协方差矩阵
    V = np.cov(data.T)
    # 对data进行主成分分析
    pca = PCA()  # create a PCA object
    # 求出主成分
    R = pca.fit_transform(data)
    # 求出主成分系数，即特征向量
    coeff = pca.components_
    # 求出主成分方差，即各特征向量对应的特征值
    latent = pca.explained_variance_
    # 求出每一个主成分所贡献的比例
    explained = pca.explained_variance_ratio_

    score = np.dot(data, coeff)
    output = pd.DataFrame(data=score,
                          index=data.index,
                          columns=map(lambda x: "第" + str(x) + "主成分", list(range(1, len(data.columns) + 1))))
    st.write("提取主成分后的数据：")
    st.dataframe(output)

    # 设置最多绘制的主成分个数
    max_pc_plotted = st.sidebar.number_input(label='最多绘制的主成分个数', value=3, step=1, min_value=1)

    # 绘制方差比例碎石图
    per_variance_ratio = np.round(explained * 100, decimals=1)
    number_of_pc_plotted = min(len(per_variance_ratio), max_pc_plotted)
    per_variance_ratio = per_variance_ratio[0:number_of_pc_plotted]
    labels = ['Principal Component' + str(x) for x in range(1, number_of_pc_plotted + 1)]
    plt.bar(x=range(1, number_of_pc_plotted + 1), height=per_variance_ratio, tick_label=labels)
    plt.ylabel('Variance Proportion(%)')
    plt.xlabel('Principal Components')
    plt.title('Scree Plot')
    st.pyplot(plt)
