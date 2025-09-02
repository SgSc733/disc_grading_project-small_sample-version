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
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='椎间盘分级系统设计')
    parser.add_argument(
        '--input', 
        type=str, 
        default=None,
        help='指定输入CSV文件路径（默认从data/raw目录自动读取）'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='指定输出目录（默认为results）'
    )
    return parser.parse_args()

def main():
    """
    主函数 - 执行完整的分级系统设计流程
    """
    args = parse_args()
    
    print("=" * 80)
    print("椎间盘分级系统设计")
    print("=" * 80)
    
    # 1. 数据加载
    print("\n[步骤1] 数据加载")
    if args.input:
        print(f"加载指定文件: {args.input}")
        df = load_features(args.input)
    else:
        print(f"从 {RAW_DATA_DIR} 目录自动加载特征文件...")

        df = load_features()
        
        print(f"成功加载数据: {df.shape[0]} 个样本, {df.shape[1]} 个特征")
        
        # 显示数据基本信息
        print(f"\n椎间盘水平分布:")
        print(df['Disc_Level'].value_counts())
    
    # 2. 数据预处理
    print("\n[步骤2] 数据预处理")
    
    # 2.1 按椎间盘水平分组（可选）
    print("是否按椎间盘水平分别处理? (y/n): ", end='')
    group_by_disc = input().strip().lower() == 'y'
    
    if group_by_disc:
        # 选择要处理的椎间盘水平
        print(f"可用的椎间盘水平: {df['Disc_Level'].unique().tolist()}")
        print("请输入要处理的椎间盘水平 (例如: L4-L5): ", end='')
        selected_disc = input().strip()
        
        if selected_disc not in df['Disc_Level'].values:
            print(f"错误: 未找到椎间盘水平 '{selected_disc}'")
            return
        
        # 筛选数据
        df_filtered = df[df['Disc_Level'] == selected_disc].copy()
        print(f"已选择 {selected_disc}，包含 {len(df_filtered)} 个样本")
    else:
        df_filtered = df.copy()
        print("处理所有椎间盘水平的数据")
    
    # 2.2 特征标准化
    print(f"\n执行特征标准化 (方法: {STANDARDIZATION_METHOD})...")
    df_standardized, scaler, feature_columns = standardize_features(
        df_filtered, 
        method=STANDARDIZATION_METHOD
    )
    print(f"已标准化 {len(feature_columns)} 个特征")
    
    # 3. PCA降维
    print(f"\n[步骤3] PCA降维 (降至 {PCA_N_COMPONENTS} 维)")
    df_pca, pca_model = apply_pca(
        df_standardized, 
        feature_columns, 
        n_components=PCA_N_COMPONENTS
    )
    
    # 显示方差贡献率
    variance_ratio = pca_model.explained_variance_ratio_
    cumsum_ratio = np.cumsum(variance_ratio)
    print(f"各主成分方差贡献率: {variance_ratio}")
    print(f"累积方差贡献率: {cumsum_ratio}")
    
    # 保存处理后的数据
    save_processed_data(df_pca)
    
    # 4. 层次聚类
    print(f"\n[步骤4] 层次聚类 (方法: {CLUSTERING_METHOD}, 距离: {CLUSTERING_METRIC})")
        
    # 获取PCA特征列
    pca_feature_columns = [col for col in df_pca.columns if col.startswith('PC')]
        
    # 执行聚类
    linkage_matrix, X_pca = perform_hierarchical_clustering(
        df_pca, 
        pca_feature_columns,
        method=CLUSTERING_METHOD,
        metric=CLUSTERING_METRIC
    )
        
    # 5. 确定聚类数量
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
        # 自动确定聚类数量
        print("\n使用肘部法则自动确定最佳聚类数量...")
        optimal_clusters = find_optimal_clusters_elbow(
            linkage_matrix, 
            X_pca,
            max_clusters=min(10, len(df_pca) // 5),  # 最大聚类数不超过样本数的1/5
            plot=True
        )
        n_clusters_to_use = optimal_clusters
        print(f"\n自动确定的聚类数量: {n_clusters_to_use}")
            
    else:
        # 手动指定聚类数量
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

    # 6. 分配聚类标签
    print(f"\n[步骤6] 分配聚类标签 (聚类数: {n_clusters_to_use})")
    
    dendrogram_path = os.path.join(FIGURES_DIR, DENDROGRAM_FILE)
    plot_dendrogram(
        linkage_matrix, 
        labels=df_pca['Sample_ID'].tolist(),
        save_path=dendrogram_path,
        n_clusters=n_clusters_to_use  
    )
    
    clusters = assign_clusters(linkage_matrix, n_clusters=n_clusters_to_use)

    # 7. 等级排序
    print("\n[步骤7] 基于距离的等级排序")
    
    # 计算聚类中心
    centers = calculate_cluster_centers(X_pca, clusters)
    
    # 对聚类进行排序
    grade_mapping, distances = rank_clusters_by_distance(centers)
    
    # 显示排序结果
    print("\n聚类到等级的映射:")
    for cluster, grade in sorted(grade_mapping.items(), key=lambda x: x[1]):
        print(f"  聚类 {cluster} -> 等级 {grade} (距离: {distances[cluster]:.4f})")
    
    # 转换聚类标签为等级
    grades = np.array([grade_mapping[c] for c in clusters])
    
    # 8. 可视化
    print("\n[步骤8] 生成可视化结果")
    
    # PCA散点图
    pca_scatter_path = os.path.join(FIGURES_DIR, PCA_SCATTER_FILE)
    plot_pca_scatter(df_pca, grades, save_path=pca_scatter_path)
    
    # 9. 保存结果
    print("\n[步骤9] 保存分级结果")
    
    # 保存分级结果
    results_df = save_grading_results(df_pca, clusters, grade_mapping)
    
    # 生成等级解释表
    if len(feature_columns) > 0:
        # 对于标准化后的特征，生成解释
        interpretation_df = generate_grade_interpretation(
            df_standardized, 
            feature_columns[:5],  # 只显示前5个特征的统计
            grades
        )
        
        interpretation_path = os.path.join(TABLES_DIR, GRADE_INTERPRETATION_FILE)
        interpretation_df.to_csv(interpretation_path, index=False)
        print(f"等级解释表已保存到: {interpretation_path}")
    
    # 10. 生成分析报告
    print("\n[步骤10] 生成分析报告")
        
    report_content = f"""
椎间盘分级系统分析报告
{'=' * 50}

1. 数据概览
   - 总样本数: {len(df_filtered)}
   - 特征数: {len(feature_columns)}
   - 椎间盘水平: {df_filtered['Disc_Level'].unique().tolist()}

2. 预处理参数
   - 标准化方法: {STANDARDIZATION_METHOD}
   - 处理的椎间盘水平: {'所有' if not group_by_disc else selected_disc}

3. PCA降维结果
   - 降维后维度: {PCA_N_COMPONENTS}
   - 各主成分方差贡献率: {variance_ratio}
   - 累积方差贡献率: {cumsum_ratio}

4. 聚类参数
   - 聚类方法: {CLUSTERING_METHOD}
   - 距离度量: {CLUSTERING_METRIC}
   - 聚类数确定方式: {'自动（肘部法则）' if choice == '1' else '手动指定'}
   - 最终聚类数: {n_clusters_to_use}
   - 配置文件默认值: {N_CLUSTERS}

5. 等级分配结果
"""
    
    # 添加每个等级的统计信息
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
   - PCA散点图: {pca_scatter_path}

{'=' * 50}
分析完成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_path = os.path.join(RESULTS_DIR, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"分析报告已保存到: {report_path}")
    
    # 保存分析参数（用于重现）
    params = {
        'standardization_method': STANDARDIZATION_METHOD,
        'pca_n_components': PCA_N_COMPONENTS,
        'clustering_method': CLUSTERING_METHOD,
        'clustering_metric': CLUSTERING_METRIC,
        'n_clusters': N_CLUSTERS,
        'processed_disc_level': 'all' if not group_by_disc else selected_disc,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    params_path = os.path.join(RESULTS_DIR, 'analysis_parameters.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"分析参数已保存到: {params_path}")
    
    print("\n" + "=" * 80)
    print("分级系统设计完成！")
    print("=" * 80)
    
    # 询问是否打开结果目录
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