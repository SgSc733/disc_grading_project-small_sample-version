import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def load_features(file_path=None):
    """
    加载特征数据
    自动从 data/raw 目录读取唯一的CSV文件
    """
    if file_path is None:
        # 自动从 data/raw 目录查找CSV文件
        raw_data_dir = RAW_DATA_DIR
        
        # 确保目录存在
        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"数据目录不存在: {raw_data_dir}")
        
        # 查找所有CSV文件
        csv_files = [f for f in os.listdir(raw_data_dir) 
                     if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"在 {raw_data_dir} 目录中未找到CSV文件。\n"
                f"请将特征文件（如 final_robust_features.csv）放置在该目录中。"
            )
        elif len(csv_files) > 1:
            print(f"在 {raw_data_dir} 中发现多个CSV文件:")
            for i, f in enumerate(csv_files, 1):
                print(f"  {i}. {f}")
            # 默认选择第一个，但给出警告
            print(f"警告: 发现多个CSV文件，将使用第一个文件: {csv_files[0]}")
            print("建议: 请确保 data/raw 目录中只有一个待处理的CSV文件\n")
            file_path = os.path.join(raw_data_dir, csv_files[0])
        else:
            # 只有一个CSV文件，直接使用
            file_path = os.path.join(raw_data_dir, csv_files[0])
            print(f"自动加载文件: {csv_files[0]}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试其他编码
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            df = pd.read_csv(file_path, encoding='latin1')
    
    # 确保包含必要的列
    required_columns = ['Sample_ID', 'Disc_Level']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    return df

def group_by_disc_level(df):
    """
    按椎间盘水平分组
    """
    grouped_data = {}
    for disc_level in DISC_GROUPS:
        mask = df['Disc_Level'] == disc_level
        grouped_data[disc_level] = df[mask].copy()
    
    return grouped_data

def group_features_by_type(feature_columns):
    """
    根据PyRadiomics命名规则对特征进行分组
    """
    shape_cols = [col for col in feature_columns if col.startswith('firstorder_')]
    intensity_cols = [col for col in feature_columns if 'firstorder_' in col or 'glcm_' in col or 'gldm_' in col or 'glrlm_' in col or 'glszm_' in col or 'ngtdm_' in col]
    texture_cols = [col for col in feature_columns if any(x in col for x in ['glcm_', 'gldm_', 'glrlm_', 'glszm_', 'ngtdm_'])]
    
    # 更精确的分组
    shape_cols = [col for col in feature_columns if 'shape_' in col.lower()]
    intensity_cols = [col for col in feature_columns if 'firstorder_' in col.lower()]
    texture_cols = [col for col in feature_columns if any(x in col.lower() for x in ['glcm_', 'gldm_', 'glrlm_', 'glszm_', 'ngtdm_'])]
    
    return {
        'shape': shape_cols,
        'intensity': intensity_cols,
        'texture': texture_cols,
        'all': feature_columns
    }

def standardize_features(df, method='zscore', feature_columns=None):
    """
    特征标准化
    """
    if feature_columns is None:
        # 自动识别特征列（排除ID和分组列）
        non_feature_cols = ['Sample_ID', 'Disc_Level', 'Patient_ID']
        feature_columns = [col for col in df.columns if col not in non_feature_cols]
    
    # 复制数据框
    df_standardized = df.copy()
    
    # 选择标准化方法
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    # 对特征进行标准化
    df_standardized[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df_standardized, scaler, feature_columns

def apply_pca(df, feature_columns, n_components=2):
    """
    应用PCA降维
    """
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(df[feature_columns])
    
    # 创建PCA特征的DataFrame
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
    
    # 合并原始ID信息
    result_df = pd.concat([df[['Sample_ID', 'Disc_Level']], pca_df], axis=1)
    
    return result_df, pca

def apply_pca_adaptive(df, feature_columns, variance_threshold=0.95):
    """
    自适应PCA降维 - 根据方差贡献率自动选择组件数
    """
    # 首先使用所有组件进行PCA
    pca_full = PCA()
    pca_full.fit(df[feature_columns])
    
    # 计算累积方差贡献率
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # 找到满足阈值的最小组件数
    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
    n_components = min(n_components, len(feature_columns))
    
    print(f"自动选择 {n_components} 个主成分（累积方差贡献率: {cumsum_variance[n_components-1]:.4f}）")
    
    # 使用选定的组件数重新进行PCA
    return apply_pca(df, feature_columns, n_components)

def save_processed_data(df, output_dir=None):
    """
    保存处理后的数据
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    output_path = os.path.join(output_dir, PROCESSED_FEATURES_FILE)
    df.to_csv(output_path, index=False)
    print(f"处理后的数据已保存到: {output_path}")
    
    return output_path