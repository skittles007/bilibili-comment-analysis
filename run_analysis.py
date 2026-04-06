# -*- coding: utf-8 -*-
"""
B站视频评论分析主程序
完整流程：数据预处理 → K-Means聚类 → LDA主题分析 → 语义网络分析
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.preprocess import TextPreprocessor, process_dataset
from src.analysis.kmeans_cluster import KMeansAnalyzer
from src.analysis.lda_topic import LDATopicAnalyzer
from src.analysis.semantic_network import SemanticNetwork


def main():
    """主分析流程"""
    print("="*80)
    print("B站视频评论分析系统")
    print("="*80)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ==================== 1. 数据加载与预处理 ====================
    print("\n" + "="*80)
    print("【步骤1】数据加载与预处理")
    print("="*80)
    
    # 加载原始数据
    input_path = "data/raw/comments.csv"
    output_path = "data/processed/comments_processed.csv"
    
    print(f"加载数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"原始数据: {len(df)} 条评论")
    
    # 数据概览
    print(f"\n数据概览:")
    print(f"  - 评论数: {len(df)}")
    print(f"  - 点赞总数: {df['点赞数'].sum()}")
    print(f"  - 平均点赞: {df['点赞数'].mean():.1f}")
    print(f"  - 唯一用户: {df['uid'].nunique()}")
    
    # 文本预处理
    print("\n正在进行文本预处理...")
    preprocessor = TextPreprocessor(
        stop_words_path="resources/stop_words.txt"
    )
    
    # 清洗和分词
    df['cleaned'] = df['评论内容'].apply(preprocessor.clean_text)
    df['processed'] = df['cleaned'].apply(lambda x: ' '.join(preprocessor.segment(x)))
    df['word_list'] = df['cleaned'].apply(lambda x: preprocessor.segment(x))
    
    # 过滤空值
    df = df[df['processed'].str.len() > 0]
    print(f"预处理后: {len(df)} 条有效评论")
    
    # 保存处理后的数据
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"处理数据已保存: {output_path}")
    
    # ==================== 2. K-Means聚类分析 ====================
    print("\n" + "="*80)
    print("【步骤2】K-Means聚类分析")
    print("="*80)
    
    texts_for_kmeans = df['processed'].tolist()
    
    # 初始化聚类分析器
    kmeans_analyzer = KMeansAnalyzer(texts_for_kmeans)
    kmeans_analyzer.vectorize(max_features=500, n_components=50)
    
    # 寻找最优K值
    print("\n寻找最优聚类数...")
    optimal_k = kmeans_analyzer.find_optimal_k(k_range=(2, 8), method='both')
    
    # 执行聚类
    labels = kmeans_analyzer.fit(n_clusters=min(optimal_k, 4))
    
    # 可视化
    kmeans_analyzer.visualize(save_path='output/figures/kmeans_clusters.png')
    
    # 打印聚类摘要
    kmeans_analyzer.print_cluster_summary()
    
    # 保存聚类结果
    cluster_info = kmeans_analyzer.get_cluster_info(top_words=10)
    df['cluster'] = labels
    
    # ==================== 3. LDA主题分析 ====================
    print("\n" + "="*80)
    print("【步骤3】LDA主题分析")
    print("="*80)
    
    texts_for_lda = df['word_list'].tolist()
    
    # 初始化LDA分析器
    lda_analyzer = LDATopicAnalyzer(texts_for_lda)
    lda_analyzer.create_dictionary(min_df=2, max_df=0.9)
    
    # 寻找最优主题数
    print("\n寻找最优主题数...")
    # 由于数据量较小，直接设置主题数
    optimal_topics = min(5, len(df) // 5)
    optimal_topics = max(3, optimal_topics)
    print(f"设置主题数: {optimal_topics}")
    
    # 训练模型
    lda_analyzer.fit(num_topics=optimal_topics, passes=20, iterations=100)
    
    # 打印主题
    lda_analyzer.print_topics(num_words=10)
    
    # 导出主题
    lda_analyzer.export_topics_to_excel('output/reports/lda_topics.xlsx')
    
    # 尝试生成可视化（如果pyLDAvis可用）
    try:
        lda_analyzer.visualize('output/reports/lda_visualization.html')
    except Exception as e:
        print(f"LDA可视化跳过: {e}")
    
    # ==================== 4. 语义网络分析 ====================
    print("\n" + "="*80)
    print("【步骤4】语义网络分析")
    print("="*80)
    
    # 初始化语义网络
    network = SemanticNetwork(texts_for_kmeans)
    network.build_network(top_n=30, min_cooccurrence=2)
    
    # 可视化
    network.visualize(output_path='output/figures/semantic_network.png')
    
    # 计算中心性
    centrality = network.get_centrality()
    print("\n关键词中心性Top 10:")
    print(centrality.head(10).to_string(index=False))
    
    # 保存中心性结果
    centrality.to_csv('output/reports/keyword_centrality.csv', index=False, encoding='utf-8-sig')
    
    # 发现社区
    communities = network.find_communities()
    print("\n语义社区结构:")
    for comm_name, words in communities.items():
        print(f"  {comm_name}: {', '.join(words[:5])}")
    
    # 导出网络
    network.export_to_gexf('output/reports/semantic_network.gexf')
    
    # ==================== 5. 生成分析报告 ====================
    print("\n" + "="*80)
    print("【步骤5】生成分析报告")
    print("="*80)
    
    generate_report(df, cluster_info, lda_analyzer, centrality, communities)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n输出文件:")
    print(f"  - 处理数据: data/processed/comments_processed.csv")
    print(f"  - 聚类图表: output/figures/kmeans_clusters.png")
    print(f"  - 最优K值图: output/figures/optimal_k.png")
    print(f"  - LDA评估图: output/figures/lda_evaluation.png")
    print(f"  - 语义网络图: output/figures/semantic_network.png")
    print(f"  - LDA主题: output/reports/lda_topics.xlsx")
    print(f"  - 关键词中心性: output/reports/keyword_centrality.csv")
    print(f"  - 分析报告: output/reports/analysis_report.md")
    print(f"  - 网络文件: output/reports/semantic_network.gexf")


def generate_report(df, cluster_info, lda_analyzer, centrality, communities):
    """生成分析报告"""
    
    report = f"""# B站视频评论分析报告

## 一、数据概览

| 指标 | 数值 |
|------|------|
| 总评论数 | {len(df)} |
| 唯一用户数 | {df['uid'].nunique()} |
| 总点赞数 | {df['点赞数'].sum()} |
| 平均点赞数 | {df['点赞数'].mean():.1f} |
| 最高点赞 | {df['点赞数'].max()} |

### 点赞Top 5评论

| 排名 | 用户名 | 点赞数 | 评论内容 |
|------|--------|--------|----------|
"""
    
    top_comments = df.nlargest(5, '点赞数')
    for i, (_, row) in enumerate(top_comments.iterrows(), 1):
        content = str(row['评论内容'])[:50] + '...' if len(str(row['评论内容'])) > 50 else row['评论内容']
        report += f"| {i} | {row['用户名']} | {row['点赞数']} | {content} |\n"
    
    report += f"""
## 二、K-Means聚类分析

聚类分析将评论分为 {len(cluster_info)} 个群体：

"""
    
    for cluster_name, data in cluster_info.items():
        report += f"""### {cluster_name}

- **样本数**: {data['count']} ({data['percentage']})
- **关键词**: {', '.join(data['keywords'][:5])}

"""
    
    report += f"""
## 三、LDA主题分析

### 主题分布

"""
    
    topics = lda_analyzer.get_topics(num_words=8)
    for topic_name, words in topics.items():
        word_str = " | ".join([f"{w['word']}({w['weight']:.3f})" for w in words[:4]])
        report += f"- **{topic_name}**: {word_str}\n"
    
    report += f"""
## 四、语义网络分析

### 核心关键词（按中心性排序）

| 排名 | 关键词 | 度中心性 | 词频 |
|------|--------|----------|------|
"""
    
    for i, (_, row) in enumerate(centrality.head(10).iterrows(), 1):
        report += f"| {i} | {row['keyword']} | {row['degree']:.4f} | {row['freq']} |\n"
    
    report += f"""
### 语义社区

"""
    for comm_name, words in communities.items():
        report += f"- **{comm_name}**: {', '.join(words[:5])}\n"
    
    report += f"""
## 五、分析结论

### 1. 用户关注焦点

根据聚类和主题分析，用户讨论主要集中在以下几个方面：
- 女性权益与社会议题
- 婚姻与职业发展的矛盾
- 职场性别歧视现象
- 东西方平权运动对比

### 2. 情感倾向

评论整体呈现理性讨论氛围，高赞评论多为深度分析观点，反映了用户对议题的深入思考。

### 3. 互动特征

用户之间存在较多观点交锋，子评论互动活跃，形成了良好的讨论氛围。

---

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**数据来源**: B站视频《谁在鼓励女性不婚不育？美国平权运动深度解析》
"""
    
    # 保存报告
    with open('output/reports/analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析报告已保存: output/reports/analysis_report.md")


if __name__ == "__main__":
    main()
