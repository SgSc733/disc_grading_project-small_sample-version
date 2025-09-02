import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 结果路径配置
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')

# 确保所有目录存在
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 数据处理参数
STANDARDIZATION_METHOD = 'robust'  
DISC_GROUPS = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']  # 椎间盘分组

# PCA参数
PCA_N_COMPONENTS = 2  # PCA降维后的维度
PCA_EXPLAINED_VARIANCE_THRESHOLD = 0.95  # 累积方差贡献率阈值

# 聚类参数
CLUSTERING_METHOD = 'ward'  # 层次聚类方法: 'ward', 'complete', 'average', 'single'
CLUSTERING_METRIC = 'euclidean'  # 距离度量: 'euclidean', 'manhattan', 'cosine'
N_CLUSTERS = 8  # 聚类数量（分级数）
DENDROGRAM_TRUNCATE_MODE = None  # 树状图截断模式: None, 'lastp', 'level'
DENDROGRAM_P = 30  # 当truncate_mode='lastp'时显示的节点数

# 可视化参数
FIGURE_DPI = 300  # 图像分辨率
FIGURE_SIZE = (10, 8)  # 图像大小
COLOR_PALETTE = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']  # 颜色方案

# 输出文件名
PROCESSED_FEATURES_FILE = 'processed_features.csv'
NEW_GRADES_FILE = 'new_grades.csv'
GRADE_INTERPRETATION_FILE = 'grade_interpretation.csv'
PCA_SCATTER_FILE = 'pca_scatter_plot.png'
DENDROGRAM_FILE = 'dendrogram.png'
