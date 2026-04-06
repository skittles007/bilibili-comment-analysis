# B站视频评论文本分析系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于Python爬虫、LDA主题分析、K-Means聚类和语义网络分析的B站视频评论深度挖掘系统。

## 项目简介

本项目通过对B站视频评论数据进行采集和深度分析，挖掘用户评论的主题、情感倾向和语义结构，为内容创作者、品牌方和平台运营者提供数据驱动的决策支持。

### 核心功能

- **数据采集**：自动化爬取B站视频评论，支持父评论和子评论
- **聚类分析**：基于TF-IDF和K-Means识别评论群体模式
- **主题分析**：LDA模型提取潜在讨论话题
- **语义网络**：构建关键词语义关系图谱
- **可视化展示**：多维度图表直观呈现分析结果

### 商业价值

| 应用场景 | 价值体现 |
|---------|---------|
| 内容优化 | 根据用户反馈调整内容策略，提升观看时长和转化率 |
| 精准营销 | 基于话题偏好制定营销策略，提高广告投放效果 |
| 用户洞察 | 了解用户需求和偏好，优化推荐算法和用户体验 |
| 趋势分析 | 识别热点话题，把握市场动向，调整产品策略 |

## 项目结构

```
bilibili-comment-analysis/
├── README.md                   # 项目说明文档
├── requirements.txt            # 依赖包列表
├── run_analysis.py             # 一键运行分析脚本
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   │   └── comments.csv        # 爬取的评论数据
│   └── processed/              # 处理后数据
│       └── comments_processed.csv
├── src/                        # 源代码目录
│   ├── crawler/                # 爬虫模块
│   │   ├── bilibili_crawler.py # B站评论爬虫
│   │   └── bv_converter.py     # BV/AV号转换
│   ├── analysis/               # 分析模块
│   │   ├── kmeans_cluster.py   # K-Means聚类分析
│   │   ├── lda_topic.py        # LDA主题分析
│   │   └── semantic_network.py # 语义网络分析
│   └── utils/                  # 工具模块
│       ├── preprocess.py       # 数据预处理
│       └── visualization.py    # 可视化工具
├── output/                     # 输出目录
│   ├── figures/                # 图表输出
│   │   ├── kmeans_clusters.png # 聚类可视化
│   │   ├── optimal_k.png       # 最优K值评估
│   │   └── semantic_network.png# 语义网络图
│   └── reports/                # 分析报告
│       ├── analysis_report.md  # 完整分析报告
│       ├── lda_topics.xlsx     # LDA主题权重
│       ├── lda_visualization.html
│       └── keyword_centrality.csv
└── resources/                  # 资源文件
    ├── stop_words.txt          # 停用词表
    └── user_dict.txt           # 自定义词典
```

## 快速开始

### 环境配置

```bash
# 克隆项目
git clone https://github.com/yourusername/bilibili-comment-analysis.git
cd bilibili-comment-analysis

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 一键运行完整分析

```bash
# 运行完整分析流程（数据预处理 → 聚类 → 主题 → 语义网络）
python run_analysis.py
```

### 单独运行各模块

```python
# 1. 数据采集
from src.crawler.bilibili_crawler import BilibiliCrawler
crawler = BilibiliCrawler()
video_url = "https://www.bilibili.com/video/BVxxxxxxxx"
crawler.crawl_comments(video_url, output_path="data/raw/comments.csv")

# 2. K-Means聚类分析
from src.analysis.kmeans_cluster import KMeansAnalyzer
cluster_analyzer = KMeansAnalyzer(comments['processed'])
cluster_analyzer.fit(n_clusters=4)
cluster_analyzer.visualize()

# 3. LDA主题分析
from src.analysis.lda_topic import LDATopicAnalyzer
lda_analyzer = LDATopicAnalyzer(comments['word_list'])
lda_analyzer.fit(n_topics=5)

# 4. 语义网络分析
from src.analysis.semantic_network import SemanticNetwork
network = SemanticNetwork(comments['processed'])
network.build_network(top_n=50)
network.visualize()
```

## 技术架构

### 1. 数据采集层

```
视频URL → BV/AV转换 → WBI签名 → API请求 → JSON解析 → CSV存储
```

**采集字段**：
- 楼层、时间、点赞数
- 用户ID、用户名、性别、等级
- 评论内容、会员等级

### 2. 数据处理层

```
原始评论 → 去重清洗 → 中文分词 → 停用词过滤 → 向量化
```

**处理流程**：
- 正则表达式过滤无效字符
- jieba分词 + 自定义词典
- 停用词表过滤
- TF-IDF权重计算

### 3. 分析建模层

| 分析方法 | 算法 | 输出 |
|---------|------|------|
| 聚类分析 | K-Means + T-SNE | 用户群体划分 |
| 主题分析 | LDA + PyLDAvis | 话题提取 |
| 语义网络 | NetworkX + Spring Layout | 关键词关系图 |

---

## 📊 分析结果示例

以下基于B站视频《谁在鼓励女性不婚不育？美国平权运动深度解析》（BV1ygcyz7EFY）的35条评论数据进行的完整分析结果。

### 数据概览

| 指标 | 数值 |
|------|------|
| 总评论数 | 35 |
| 唯一用户数 | 29 |
| 总点赞数 | 27,223 |
| 平均点赞数 | 777.8 |
| 最高点赞 | 15,511 |

### K-Means聚类分析

基于手肘法和轮廓系数确定最优聚类数为4：

| 簇编号 | 样本数 | 占比 | 关键词 | 主题特征 |
|--------|--------|------|--------|----------|
| 簇0 | 8 | 22.9% | 一辈, 一口, peterson, 丈夫 | 家庭观念讨论 |
| 簇1 | 8 | 22.9% | jordan, 丁克, 一种, 下面 | 丁克与个人选择 |
| 簇2 | 5 | 14.3% | 一大批, 一旦, 一段话, 三年 | 现象观察分析 |
| 簇3 | 14 | 40.0% | 一系列, 不会, 一口 | 综合性讨论 |

**聚类可视化**：

![K-Means聚类结果](output/figures/kmeans_clusters.png)

### LDA主题分析

最优主题数为5，各主题关键词分布如下：

| 主题 | 关键词 | 主题描述 |
|------|--------|----------|
| 主题1 | 结婚(0.090) 问题(0.062) 男女(0.034) | 婚姻与性别议题 |
| 主题2 | 企业(0.086) 社会(0.086) 对立(0.047) | 企业与社会结构 |
| 主题3 | 年轻人(0.071) 家务(0.071) 结婚(0.071) | 年轻人的生活选择 |
| 主题4 | 结婚(0.091) 女性(0.047) 稳定(0.047) | 女性婚姻观 |
| 主题5 | 晚点(0.058) 企业(0.058) 问题(0.058) | 职业发展与婚姻 |

### 语义网络分析

#### 核心关键词中心性

| 排名 | 关键词 | 度中心性 | 词频 |
|------|--------|----------|------|
| 1 | 结婚 | 0.2759 | 18 |
| 2 | 家务 | 0.1724 | 6 |
| 3 | 年轻人 | 0.1379 | 8 |
| 4 | 现在 | 0.1379 | 7 |
| 5 | 不想 | 0.1034 | 8 |

**语义网络可视化**：

![语义网络图](output/figures/semantic_network.png)

### 分析结论

#### 1. 用户关注焦点

根据聚类和主题分析，用户讨论主要集中在以下几个方面：
- **女性权益与社会议题**：平权运动、性别对立
- **婚姻与职业发展的矛盾**：晚婚、丁克现象
- **职场性别歧视现象**：企业对女性员工的态度
- **家庭劳动分工**：家务、育儿责任分配

#### 2. 情感倾向

评论整体呈现理性讨论氛围，高赞评论多为深度分析观点，反映了用户对议题的深入思考。点赞Top 5评论均超过500赞，说明用户对视频内容的共鸣较强。

#### 3. 互动特征

用户之间存在较多观点交锋，子评论互动活跃，形成了良好的讨论氛围。唯一用户占比82.9%（29/35），说明讨论较为分散，观点多元。

---

## 依赖说明

```
# 核心依赖
requests>=2.28.0
pandas>=1.5.0
numpy>=1.23.0
jieba>=0.42.1

# 机器学习
scikit-learn>=1.2.0
gensim>=4.3.0

# 可视化
matplotlib>=3.6.0
seaborn>=0.12.0
pyLDAvis>=3.4.0
networkx>=2.8.0

# 工具
tqdm>=4.64.0
```

## 注意事项

1. **爬虫使用**：请遵守B站robots协议，合理设置请求间隔
2. **数据安全**：敏感数据请勿上传至公开仓库
3. **分析结果**：主题数和聚类数需根据具体数据调整
4. **字体支持**：如需生成中文图表，请确保系统安装中文字体

## 更新日志

- **v1.1.0** (2024-08-08): 添加完整分析示例
  - 新增 `run_analysis.py` 一键运行脚本
  - 完善README分析结果展示
  - 添加语义网络分析输出
- **v1.0.0** (2024-08-08): 初始版本发布
  - 完成B站评论爬虫功能
  - 实现K-Means聚类分析
  - 实现LDA主题分析
  - 实现语义网络分析

## 参考来源

本项目基于阿里云开发者社区文章[《基于B站视频评论的文本分析》](https://developer.aliyun.com/article/1579624)进行整理和复现。

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

**Star** ⭐ 本项目，获取更多文本分析实战案例！