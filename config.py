import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')

for dir_path in [RAW_DATA_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

STANDARDIZATION_METHOD = 'robust'  
DISC_GROUPS = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

PCA_N_COMPONENTS = 2
PCA_EXPLAINED_VARIANCE_THRESHOLD = 0.95

CLUSTERING_METHOD = 'ward'
CLUSTERING_METRIC = 'euclidean'
N_CLUSTERS = 8
DENDROGRAM_TRUNCATE_MODE = None
DENDROGRAM_P = 30

FIGURE_DPI = 300
FIGURE_SIZE = (10, 8)
COLOR_PALETTE = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

PROCESSED_FEATURES_FILE = 'processed_features.csv'
NEW_GRADES_FILE = 'new_grades.csv'
GRADE_INTERPRETATION_FILE = 'grade_interpretation.csv'
PCA_SCATTER_FILE = 'pca_scatter_plot.png'
DENDROGRAM_FILE = 'dendrogram.png'

CLINICAL_COLUMNS_CANDIDATES = ['Pain_Score', 'ODI_Score', 'VAS_Score', 'JOA_Score']
