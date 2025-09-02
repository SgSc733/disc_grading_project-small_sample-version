import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def load_features(file_path=None):

    if file_path is None:
        raw_data_dir = RAW_DATA_DIR
        
        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"数据目录不存在: {raw_data_dir}")
        
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
            print(f"警告: 发现多个CSV文件，将使用第一个文件: {csv_files[0]}")
            print("建议: 请确保 data/raw 目录中只有一个待处理的CSV文件\n")
            file_path = os.path.join(raw_data_dir, csv_files[0])
        else:
            file_path = os.path.join(raw_data_dir, csv_files[0])
            print(f"自动加载文件: {csv_files[0]}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            df = pd.read_csv(file_path, encoding='latin1')
    
    required_columns = ['Sample_ID', 'Disc_Level']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    return df

def group_by_disc_level(df):

    grouped_data = {}
    for disc_level in DISC_GROUPS:
        mask = df['Disc_Level'] == disc_level
        grouped_data[disc_level] = df[mask].copy()
    
    return grouped_data

def group_features_by_type(feature_columns):

    shape_cols = [col for col in feature_columns if col.startswith('firstorder_')]
    intensity_cols = [col for col in feature_columns if 'firstorder_' in col or 'glcm_' in col or 'gldm_' in col or 'glrlm_' in col or 'glszm_' in col or 'ngtdm_' in col]
    texture_cols = [col for col in feature_columns if any(x in col for x in ['glcm_', 'gldm_', 'glrlm_', 'glszm_', 'ngtdm_'])]
    
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

    if feature_columns is None:
        non_feature_cols = ['Sample_ID', 'Disc_Level', 'Patient_ID']
        feature_columns = [col for col in df.columns if col not in non_feature_cols]
    
    df_standardized = df.copy()
    
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    df_standardized[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df_standardized, scaler, feature_columns

def apply_pca(df, feature_columns, n_components=2):

    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(df[feature_columns])
    
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
    
    result_df = pd.concat([df[['Sample_ID', 'Disc_Level']], pca_df], axis=1)
    
    return result_df, pca

def apply_pca_adaptive(df, feature_columns, variance_threshold=0.95):

    pca_full = PCA()
    pca_full.fit(df[feature_columns])
    
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
    n_components = min(n_components, len(feature_columns))
    
    print(f"自动选择 {n_components} 个主成分（累积方差贡献率: {cumsum_variance[n_components-1]:.4f}）")
    
    return apply_pca(df, feature_columns, n_components)

def save_processed_data(df, output_dir=None):

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    output_path = os.path.join(output_dir, PROCESSED_FEATURES_FILE)
    df.to_csv(output_path, index=False)
    print(f"处理后的数据已保存到: {output_path}")
    
    return output_path
