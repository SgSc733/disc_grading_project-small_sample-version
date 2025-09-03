import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def load_and_filter_features(raw_features_path, robust_features_list_path, clinical_data_path=None):

    print(f"  - 正在加载原始特征总表: {os.path.basename(raw_features_path)}")
    try:
        df_raw = pd.read_csv(raw_features_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_raw = pd.read_csv(raw_features_path, encoding='gbk')
        except:
            df_raw = pd.read_csv(raw_features_path, encoding='latin1')
    
    id_col_name = df_raw.columns[0]
    print(f"  - 识别样本ID列: '{id_col_name}'")
    
    df_raw = df_raw.rename(columns={id_col_name: 'Sample_ID'})
    
    if 'Disc_Level' not in df_raw.columns:
        df_raw['Disc_Level'] = 'L4-L5'
        print(f"  - 添加默认Disc_Level列（用于兼容）")

    print(f"  - 正在加载稳健特征清单: {os.path.basename(robust_features_list_path)}")
    df_robust_list = pd.read_csv(robust_features_list_path)

    if 'feature' not in df_robust_list.columns:
        raise ValueError("稳健特征清单文件中必须包含名为 'feature' 的列。")
    
    robust_feature_names = df_robust_list['feature'].tolist()
    
    final_columns_to_keep = ['Sample_ID', 'Disc_Level']
    
    missing_features = []
    found_features = []
    for feature in robust_feature_names:
        if feature in df_raw.columns:
            final_columns_to_keep.append(feature)
            found_features.append(feature)
        else:
            missing_features.append(feature)
    
    if missing_features:
        print(f"\n警告: 在原始特征总表中找不到以下 {len(missing_features)} 个稳健特征:")
        print(f"  {', '.join(missing_features[:5])}{'...' if len(missing_features) > 5 else ''}")
    
    print(f"\n  - 成功匹配 {len(found_features)} / {len(robust_feature_names)} 个稳健特征。")
    
    df_filtered = df_raw[final_columns_to_keep].copy()
    
    if clinical_data_path and os.path.exists(clinical_data_path):
        print(f"\n  - 正在加载临床数据: {os.path.basename(clinical_data_path)}")
        try:
            df_clinical = pd.read_csv(clinical_data_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_clinical = pd.read_csv(clinical_data_path, encoding='gbk')
            except:
                df_clinical = pd.read_csv(clinical_data_path, encoding='latin1')
        
        clinical_id_col = df_clinical.columns[0]
        print(f"  - 临床数据ID列: '{clinical_id_col}'")
        
        df_clinical = df_clinical.rename(columns={clinical_id_col: 'Sample_ID'})
        
        clinical_numeric_cols = []
        for col in df_clinical.columns:
            if col != 'Sample_ID':
                try:
                    df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')
                    if df_clinical[col].notna().sum() > 0:
                        clinical_numeric_cols.append(col)
                except:
                    pass
        
        if clinical_numeric_cols:
            print(f"  - 检测到数值型临床特征: {', '.join(clinical_numeric_cols)}")
            
            df_merged = pd.merge(df_filtered, 
                                df_clinical[['Sample_ID'] + clinical_numeric_cols], 
                                on='Sample_ID', 
                                how='left')
            
            matched_samples = df_merged[clinical_numeric_cols[0]].notna().sum()
            print(f"  - 成功匹配 {matched_samples}/{len(df_filtered)} 个样本的临床数据")
            
            df_filtered = df_merged
        else:
            print("  - 警告: 临床数据文件中未找到有效的数值型特征")
    
    print(f"\n最终数据框架:")
    print(f"  - 样本数: {len(df_filtered)}")
    print(f"  - 总列数: {len(df_filtered.columns)}")
    print(f"  - 稳健特征数: {len(found_features)}")
    
    potential_clinical_cols = ['VAS_Score', 'ODI_Score', 'ODI_Index', 'Pain_Score', 'JOA_Score']
    available_clinical = [col for col in potential_clinical_cols if col in df_filtered.columns]
    if available_clinical:
        print(f"  - 可用临床指标: {', '.join(available_clinical)}")
    
    return df_filtered

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
                f"请将特征文件放置在该目录中。"
            )
        file_path = os.path.join(raw_data_dir, csv_files[0])
    
    df = pd.read_csv(file_path)
    return df

def group_by_disc_level(df):
    grouped_data = {}
    for disc_level in DISC_GROUPS:
        mask = df['Disc_Level'] == disc_level
        grouped_data[disc_level] = df[mask].copy()
    
    return grouped_data

def group_features_by_type(feature_columns):

    shape_cols = [col for col in feature_columns if col.startswith('shape_')]
    firstorder_cols = [col for col in feature_columns if col.startswith('firstorder_')]
    glcm_cols = [col for col in feature_columns if col.startswith('glcm_')]
    gldm_cols = [col for col in feature_columns if col.startswith('gldm_')]
    glrlm_cols = [col for col in feature_columns if col.startswith('glrlm_')]
    glszm_cols = [col for col in feature_columns if col.startswith('glszm_')]
    ngtdm_cols = [col for col in feature_columns if col.startswith('ngtdm_')]
    
    texture_cols = glcm_cols + gldm_cols + glrlm_cols + glszm_cols + ngtdm_cols
    
    return {
        'shape': shape_cols,
        'firstorder': firstorder_cols,
        'texture': texture_cols,
        'all': feature_columns
    }

def standardize_features(df, method='zscore', feature_columns=None):
    if feature_columns is None:
        non_feature_cols = ['Sample_ID', 'Disc_Level', 'Patient_ID', 
                        'VAS_Score', 'ODI_Score', 'ODI_Index', 'Pain_Score', 
                        'JOA_Score', 'Clinical_Score']
        non_feature_cols = [col for col in non_feature_cols if col in df.columns]
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
