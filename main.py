import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data_utils import *
from src.clustering_utils import *
from config import *

def parse_args():
    import os
    
    parser = argparse.ArgumentParser(description='椎间盘分级系统设计')
    
    raw_features_file = None
    robust_list_file = None
    clinical_file = None
    
    if os.path.exists('data/raw'):
        csv_files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
        print(f"在 data/raw 目录找到 {len(csv_files)} 个CSV文件")
        
        for f in csv_files:
            full_path = os.path.join('data/raw', f)
            file_lower = f.lower()
            
            if 'robust' in file_lower:
                robust_list_file = full_path
                print(f"  - 识别为稳健特征清单: {f}")
            elif 'clinical' in file_lower:
                clinical_file = full_path
                print(f"  - 识别为临床数据文件: {f}")
            elif not raw_features_file:
                raw_features_file = full_path
                print(f"  - 识别为原始特征文件: {f}")
        
        potential_raw_files = []
        for f in csv_files:
            file_lower = f.lower()
            if 'robust' not in file_lower and 'clinical' not in file_lower:
                potential_raw_files.append(os.path.join('data/raw', f))
        
        if potential_raw_files:
            raw_features_file = max(potential_raw_files, key=lambda x: os.path.getsize(x))
            print(f"  - 选择最大文件作为原始特征文件: {os.path.basename(raw_features_file)}")
    
    parser.add_argument(
        '--raw-features',
        type=str,
        required=False,
        default=raw_features_file,
        help='指定包含所有特征的原始CSV文件路径'
    )
    parser.add_argument(
        '--robust-list',
        type=str,
        required=False,
        default=robust_list_file,
        help='指定稳健特征清单CSV文件路径'
    )
    parser.add_argument(
        '--clinical-data',
        type=str,
        required=False,
        default=clinical_file,
        help='指定临床数据CSV文件路径（可选）'
    )
    parser.add_argument(
        '--clinical-col',
        type=str,
        default='VAS_Score',
        help='用于临床排序的列名'
    )
    parser.add_argument(
        '--use-clinical',
        action='store_true',
        help='是否使用临床数据进行等级排序'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='指定输出目录（默认为results）'
    )
    
    args = parser.parse_args()
    
    if not args.raw_features or not os.path.exists(args.raw_features):
        print("\n错误：找不到原始特征文件！")
        print("请确保在 data/raw/ 目录中有CSV文件")
        print("或使用命令行参数: --raw-features <文件路径>")
        sys.exit(1)
    
    if not args.robust_list or not os.path.exists(args.robust_list):
        print("\n错误：找不到稳健特征清单文件！")
        print("请确保在 data/raw/ 目录中有包含'robust'的CSV文件")
        print("或使用命令行参数: --robust-list <文件路径>")
        sys.exit(1)
    
    print(f"\n使用的文件：")
    print(f"  原始特征: {os.path.basename(args.raw_features)}")
    print(f"  稳健清单: {os.path.basename(args.robust_list)}")
    
    if args.clinical_data and os.path.exists(args.clinical_data):
        print(f"  临床数据: {os.path.basename(args.clinical_data)}")
    else:
        if args.clinical_data:
            print(f"  临床数据: 未找到文件 {args.clinical_data}")
        else:
            print(f"  临床数据: 未提供")
    
    print("-" * 80)
    
    return args

def main():
    args = parse_args()
    
    print("=" * 80)
    print("椎间盘分级系统设计")
    print("=" * 80)
    
    print("\n[步骤1] 数据加载与特征筛选")
    
    from src.data_utils import load_and_filter_features
    df = load_and_filter_features(args.raw_features, args.robust_list, args.clinical_data)
    
    print(f"成功加载数据: {df.shape[0]} 个样本, {df.shape[1]} 列")
    

    clinical_columns = ['VAS_Score', 'ODI_Score', 'ODI_Index', 'Pain_Score', 
                    'JOA_Score', 'Clinical_Score']
    available_clinical = [col for col in clinical_columns if col in df.columns]
    if available_clinical:
        print(f"检测到临床数据列: {', '.join(available_clinical)}")
        if args.use_clinical:
            print(f"将使用 '{args.clinical_col}' 进行临床验证和排序")
    
    
    print("\n[步骤2] 数据预处理")
    
    df_filtered = df.copy()
    print(f"处理所有样本数据 (共 {len(df_filtered)} 个样本)")
    
    print(f"\n执行特征标准化 (方法: {STANDARDIZATION_METHOD})...")

    clinical_cols = [col for col in CLINICAL_COLUMNS_CANDIDATES if col in df_filtered.columns]

    non_feature_cols = ['Sample_ID'] + clinical_cols

    feature_columns = [col for col in df_filtered.columns if col not in non_feature_cols and np.issubdtype(df_filtered[col].dtype, np.number)]
    
    df_standardized = df_filtered.copy()

    if STANDARDIZATION_METHOD == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    elif STANDARDIZATION_METHOD == 'zscore':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    if feature_columns:
        print(f"  正在对 {len(feature_columns)} 个特征进行标准化...")
        df_standardized[feature_columns] = scaler.fit_transform(df_filtered[feature_columns])
        print("  标准化完成。")
    else:
        print("  警告：未找到可进行标准化的数值型特征。")
    
    print(f"\n[步骤3] PCA降维 (降至 {PCA_N_COMPONENTS} 维)")
    df_pca, pca_model = apply_pca(
        df_standardized, 
        feature_columns, 
        n_components=PCA_N_COMPONENTS
    )
    
    variance_ratio = pca_model.explained_variance_ratio_
    cumsum_ratio = np.cumsum(variance_ratio)
    print(f"各主成分方差贡献率: {variance_ratio}")
    
    
    print(f"\n[步骤4] 层次聚类 (方法: {CLUSTERING_METHOD}, 距离: {CLUSTERING_METRIC})")
        
    pca_feature_columns = [col for col in df_pca.columns if col.startswith('PC')]
        
    linkage_matrix, X_pca = perform_hierarchical_clustering(
        df_pca, 
        pca_feature_columns,
        method=CLUSTERING_METHOD,
        metric=CLUSTERING_METRIC
    )
        
    print("\n[步骤5] 确定聚类数量")
    print("请选择聚类数量确定方式:")
    print("  1. 自动确定（使用肘部法则）")
    print("  2. 手动指定")
        
    while True:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice in ['1', '2']:
            break
        print("无效输入，请输入 1 或 2")
        
    if choice == '1':
        print("\n使用肘部法则自动确定最佳聚类数量...")
        optimal_clusters = find_optimal_clusters_elbow(
            linkage_matrix, 
            X_pca,
            max_clusters=min(10, len(df_pca) // 5),
            plot=True
        )
        n_clusters_to_use = optimal_clusters
            
    else:
        print(f"\n默认聚类数量: {N_CLUSTERS}")
        print(f"有效范围: 2 到 {min(10, len(df_pca)-1)}")
            
        while True:
            try:
                user_input = input(f"请输入聚类数量 (直接回车使用默认值 {N_CLUSTERS}): ").strip()
                
                if user_input == "":
                    n_clusters_to_use = N_CLUSTERS
                    break
                
                n_clusters_to_use = int(user_input)
                
                if 2 <= n_clusters_to_use <= min(10, len(df_pca)-1):
                    break
                else:
                    print(f"聚类数量必须在 2 到 {min(10, len(df_pca)-1)} 之间")
            except ValueError:
                print("请输入有效的整数")
        
        print(f"\n使用手动指定的聚类数量: {n_clusters_to_use}")

    print(f"\n[步骤6] 分配聚类标签 (聚类数: {n_clusters_to_use})")
    
    dendrogram_path = os.path.join(FIGURES_DIR, DENDROGRAM_FILE)
    plot_dendrogram(
        linkage_matrix, 
        labels=df_pca['Sample_ID'].tolist(),
        save_path=dendrogram_path,
        n_clusters=n_clusters_to_use  
    )
    
    clusters = assign_clusters(linkage_matrix, n_clusters=n_clusters_to_use)

    print("\n[步骤7] 等级排序")
    
    df_with_clusters = df_filtered.copy()
    df_with_clusters['Cluster'] = clusters
    
    for col in df_pca.columns:
        if col not in ['Sample_ID', 'Disc_Level']:
            df_with_clusters[col] = df_pca[col].values
    
    from src.clustering_utils import rank_clusters_clinically
    
    available_clinical = [col for col in CLINICAL_COLUMNS_CANDIDATES if col in df_with_clusters.columns]
    
    if available_clinical and args.use_clinical:
        clinical_col_to_use = args.clinical_col if args.clinical_col in available_clinical else available_clinical[0]
        print(f"检测到临床数据，使用 '{clinical_col_to_use}' 进行排序...")
        grade_mapping, cluster_stats = rank_clusters_clinically(
            df_with_clusters,
            cluster_col='Cluster',
            clinical_col=clinical_col_to_use,
            use_clinical=True
        )
    else:
        if not available_clinical:
            print("未检测到临床数据，使用PC1位置进行排序...")
        else:
            print("检测到临床数据但未启用，使用PC1位置进行排序...")
        
        grade_mapping, cluster_stats = rank_clusters_clinically(
            df_with_clusters,
            cluster_col='Cluster',
            clinical_col=None,
            use_clinical=False
        )
    
    grades = np.array([grade_mapping[c] for c in clusters])
    
    print("\n[步骤8] 生成可视化结果")
    
    pca_scatter_path = os.path.join(FIGURES_DIR, PCA_SCATTER_FILE)

    actual_pca_scatter_path = plot_pca_scatter(df_pca, grades, save_path=pca_scatter_path)
    if actual_pca_scatter_path is None:
        actual_pca_scatter_path = "未生成 (PCA维度不足)"
    
    print("\n[步骤9] 保存分级结果")

    df_with_clusters['Grade'] = grades
    results_df = save_grading_results(df_with_clusters, clusters)
    
    if len(feature_columns) > 0:
        interpretation_df = generate_grade_interpretation(
            df_standardized, 
            feature_columns[:5],
            grades
        )
        
        interpretation_path = os.path.join(TABLES_DIR, GRADE_INTERPRETATION_FILE)
        interpretation_df.to_csv(interpretation_path, index=False)
        print(f"等级解释表已保存到: {interpretation_path}")
    
    print("\n[步骤10] 生成分析报告")
        
    report_content = f"""
椎间盘分级系统分析报告
{'=' * 50}

1. 数据概览
   - 总样本数: {len(df_filtered)}
   - 特征数: {len(feature_columns)}

2. 预处理参数
   - 标准化方法: {STANDARDIZATION_METHOD}

3. PCA降维结果
   - 降维后维度: {PCA_N_COMPONENTS}
   - 各主成分方差贡献率: {variance_ratio}

4. 聚类参数
   - 聚类方法: {CLUSTERING_METHOD}
   - 距离度量: {CLUSTERING_METRIC}
   - 聚类数确定方式: {'自动（肘部法则）' if choice == '1' else '手动指定'}
   - 最终聚类数: {n_clusters_to_use}
   - 配置文件默认值: {N_CLUSTERS}

5. 等级分配结果
"""
    
    for grade in sorted(np.unique(grades)):
        count = np.sum(grades == grade)
        samples = results_df[results_df['Grade'] == grade]['Sample_ID'].tolist()
        report_content += f"\n   等级 {grade}: {count} 个样本"
        report_content += f"\n   样本ID: {', '.join(samples[:5])}{'...' if len(samples) > 5 else ''}"
    
    report_content += f"""

6. 输出文件
   - 处理后数据: {os.path.join(PROCESSED_DATA_DIR, PROCESSED_FEATURES_FILE)}
   - 分级结果: {os.path.join(TABLES_DIR, NEW_GRADES_FILE)}
   - 等级解释: {os.path.join(TABLES_DIR, GRADE_INTERPRETATION_FILE)}
   - 树状图: {dendrogram_path}
   - PCA散点图: {actual_pca_scatter_path}

{'=' * 50}
分析完成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_path = os.path.join(RESULTS_DIR, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"分析报告已保存到: {report_path}")
    
    print("\n" + "=" * 80)
    print("分级系统设计完成！")
    print("=" * 80)
    
    print("\n是否打开结果目录? (y/n): ", end='')
    if input().strip().lower() == 'y':
        import subprocess
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', RESULTS_DIR])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', RESULTS_DIR])
        else:
            subprocess.Popen(['xdg-open', RESULTS_DIR])

if __name__ == "__main__":
    main()
