import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def perform_hierarchical_clustering(df, feature_columns, method='ward', metric='euclidean'):
    X = df[feature_columns].values
    
    if method == 'ward' and metric != 'euclidean':
        print(f"警告: Ward方法只支持欧氏距离，将使用euclidean替代{metric}")
        metric = 'euclidean'
    
    linkage_matrix = linkage(X, method=method, metric=metric)
    
    return linkage_matrix, X

def plot_dendrogram(linkage_matrix, labels=None, save_path=None, n_clusters=None):
    plt.figure(figsize=FIGURE_SIZE)
    
    if n_clusters and n_clusters > 1:
        color_threshold = linkage_matrix[-(n_clusters-1), 2] if n_clusters <= len(linkage_matrix) else None
    else:
        color_threshold = 0.7 * max(linkage_matrix[:, 2])
    
    dendrogram(
        linkage_matrix,
        labels=labels,
        truncate_mode=DENDROGRAM_TRUNCATE_MODE,
        p=DENDROGRAM_P if DENDROGRAM_TRUNCATE_MODE == 'lastp' else 0,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=color_threshold,
        above_threshold_color='#BCBCBC'
    )
    
    if n_clusters and color_threshold:
        plt.axhline(y=color_threshold, c='red', linestyle='--', linewidth=1, 
                   label=f'Cut for {n_clusters} clusters')
        plt.legend(loc='upper right')
    
    plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
    plt.xlabel('Sample ID', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"树状图已保存到: {save_path}")
    
    plt.close()

def assign_clusters(linkage_matrix, n_clusters=3):
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    return clusters

def find_optimal_clusters_elbow(linkage_matrix, X, max_clusters=10, plot=True):
    from sklearn.metrics import silhouette_score
    
    max_clusters = min(max_clusters, len(X) - 1)
    
    K = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    
    for k in K:
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        
        wss = 0
        for i in range(1, k + 1):
            cluster_mask = clusters == i
            if np.sum(cluster_mask) > 0:
                cluster_points = X[cluster_mask]
                cluster_center = np.mean(cluster_points, axis=0)
                wss += np.sum((cluster_points - cluster_center) ** 2)
        inertias.append(wss)
        
        if k < len(X):
            silhouette_avg = silhouette_score(X, clusters)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    if len(inertias) > 2:
        diff1 = np.diff(inertias)
        diff2 = np.diff(diff1)
        
        if len(diff2) > 0:
            elbow_idx = np.argmax(np.abs(diff2)) + 1
            optimal_k_elbow = K[elbow_idx]
        else:
            optimal_k_elbow = 5
    else:
        optimal_k_elbow = 5

    
    print(f"\n肘部法则建议: {optimal_k_elbow} 个聚类")
    
    if plot and len(inertias) > 0:
        plt.figure(figsize=(10, 6))
        
        plt.plot(K, inertias, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k_elbow, color='r', linestyle='--', 
                   linewidth=2, label=f'Elbow point: k={optimal_k_elbow}')
        
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Within-cluster Sum of Squares (WSS)', fontsize=12)
        plt.title('Elbow Method for Optimal Cluster Number', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        elbow_idx = K.index(optimal_k_elbow) if optimal_k_elbow in K else 0
        plt.annotate(f'Suggested: k={optimal_k_elbow}', 
                    xy=(optimal_k_elbow, inertias[elbow_idx]),
                    xytext=(optimal_k_elbow + 0.5, inertias[elbow_idx]),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
        
        plt.tight_layout()
        
        elbow_plot_path = os.path.join(FIGURES_DIR, 'elbow_analysis.png')
        plt.savefig(elbow_plot_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"\n肘部分析图已保存到: {elbow_plot_path}")
        plt.close()
    
    return optimal_k_elbow

def calculate_cluster_centers(X, clusters):
    unique_clusters = np.unique(clusters)
    centers = {}
    
    for cluster in unique_clusters:
        mask = clusters == cluster
        centers[cluster] = np.mean(X[mask], axis=0)
    
    return centers

def rank_clusters_by_distance(centers, reference_point=None):

    pc1_positions = {}
    for cluster, center in centers.items():
        pc1_positions[cluster] = center[0]
    
    sorted_clusters = sorted(pc1_positions.keys(), key=lambda x: pc1_positions[x])
    
    grade_mapping = {}
    for i, cluster in enumerate(sorted_clusters):
        grade_mapping[cluster] = i + 1
    
    print("\n基于PC1位置的聚类排序:")
    for cluster in sorted_clusters:
        print(f"  聚类 {cluster}: PC1 = {pc1_positions[cluster]:.4f} -> 等级 {grade_mapping[cluster]}")
    
    return grade_mapping, pc1_positions


def rank_clusters_clinically(df_with_clusters_and_clinical, cluster_col='Cluster', 
                            clinical_col='Pain_Score', use_clinical=True):

    from scipy import stats
    
    if use_clinical and clinical_col in df_with_clusters_and_clinical.columns:
        print(f"\n[临床整合] 基于临床指标 '{clinical_col}' 进行等级排序和验证")
        
        df_clean = df_with_clusters_and_clinical.dropna(subset=[clinical_col])
        
        if len(df_clean) < len(df_with_clusters_and_clinical):
            print(f"  警告: {len(df_with_clusters_and_clinical) - len(df_clean)} 个样本因临床数据缺失被排除")
        
        cluster_stats = df_clean.groupby(cluster_col)[clinical_col].agg(['mean', 'std', 'count'])
        cluster_stats = cluster_stats.sort_values('mean')
        
        grade_mapping = {cluster: i + 1 for i, cluster in enumerate(cluster_stats.index)}
        
        clusters = df_clean[cluster_col].unique()
        if len(clusters) > 1:
            grouped_data = [df_clean[df_clean[cluster_col] == c][clinical_col].values 
                           for c in clusters]
            
            grouped_data = [g for g in grouped_data if len(g) >= 3]
            
            if len(grouped_data) > 1:
                normality_tests = []
                for g in grouped_data:
                    if len(g) >= 3:
                        _, p = stats.shapiro(g)
                        normality_tests.append(p > 0.05)
                
                is_normal = len(normality_tests) > 0 and all(normality_tests)
                
                if is_normal:
                    stat_val, p_val = stats.f_oneway(*grouped_data)
                    test_name = "ANOVA"
                else:
                    stat_val, p_val = stats.kruskal(*grouped_data)
                    test_name = "Kruskal-Wallis"
                    
                print(f"  - 组间差异显著性检验 ({test_name}): 统计量 = {stat_val:.4f}, P-value = {p_val:.4f}")
                
                if p_val < 0.05:
                    print("  ✓ 结论: 新的影像学分组在临床指标上存在显著差异 (p<0.05)")
                elif p_val < 0.1:
                    print("  ! 结论: 新的影像学分组在临床指标上存在边缘显著差异 (p<0.1)")
                else:
                    print("  × 结论: 新的影像学分组在临床指标上未显示出显著差异 (p>0.1)")
        
        print("\n临床排序后的聚类到等级映射:")
        for cluster, row in cluster_stats.iterrows():
            grade = grade_mapping[cluster]
            print(f"  聚类 {cluster} -> 等级 {grade}")
            print(f"    平均{clinical_col}: {row['mean']:.2f} ± {row['std']:.2f} (N={int(row['count'])})")
            
        return grade_mapping, cluster_stats
    
    else:
        print("\n基于PC1位置进行等级排序")
        
        pca_columns = [col for col in df_with_clusters_and_clinical.columns if col.startswith('PC')]
        
        if pca_columns:
            cluster_pc1_means = df_with_clusters_and_clinical.groupby(cluster_col)[pca_columns[0]].mean()
            
            sorted_clusters = sorted(cluster_pc1_means.index, key=lambda x: cluster_pc1_means[x])
            
            grade_mapping = {cluster: i + 1 for i, cluster in enumerate(sorted_clusters)}
            
            print("\n基于PC1位置的聚类排序:")
            for cluster in sorted_clusters:
                print(f"  聚类 {cluster}: PC1 = {cluster_pc1_means[cluster]:.4f} -> 等级 {grade_mapping[cluster]}")
            
            return grade_mapping, None
        else:
            unique_clusters = sorted(df_with_clusters_and_clinical[cluster_col].unique())
            grade_mapping = {cluster: i + 1 for i, cluster in enumerate(unique_clusters)}
            print("\n使用默认顺序排序")
            return grade_mapping, None


def plot_pca_scatter(df, cluster_labels, save_path=None):
    from mpl_toolkits.mplot3d import Axes3D
    
    pca_columns = [col for col in df.columns if col.startswith('PC')]
    
    if len(pca_columns) < 2:
        print("警告: PCA特征不足，无法绘制散点图")
        return
    
    is_3d = len(pca_columns) >= 3
    
    if is_3d:
        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            df[pca_columns[0]], 
            df[pca_columns[1]], 
            df[pca_columns[2]],
            c=cluster_labels, 
            cmap='viridis',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        for idx, row in df.iterrows():
            ax.text(
                row[pca_columns[0]], 
                row[pca_columns[1]], 
                row[pca_columns[2]],
                row['Sample_ID'],
                fontsize=8,
                alpha=0.7
            )
        
        ax.set_xlabel(f'{pca_columns[0]}', fontsize=12)
        ax.set_ylabel(f'{pca_columns[1]}', fontsize=12)
        ax.set_zlabel(f'{pca_columns[2]}', fontsize=12)
        ax.set_title('PCA 3D Scatter Plot with Cluster Assignments', fontsize=16)
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Grade', fontsize=12)
        
        ax.view_init(elev=20, azim=45)
        
        ax.grid(True, alpha=0.3)
        
    else:
        plt.figure(figsize=FIGURE_SIZE)
        
        scatter = plt.scatter(
            df[pca_columns[0]], 
            df[pca_columns[1]], 
            c=cluster_labels, 
            cmap='viridis',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        for idx, row in df.iterrows():
            plt.annotate(
                row['Sample_ID'], 
                (row[pca_columns[0]], row[pca_columns[1]]),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        plt.xlabel(f'{pca_columns[0]}', fontsize=12)
        plt.ylabel(f'{pca_columns[1]}', fontsize=12)
        plt.title('PCA 2D Scatter Plot with Cluster Assignments', fontsize=16)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Grade', fontsize=12)
        
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        base_name = os.path.splitext(save_path)[0]
        ext = os.path.splitext(save_path)[1]
        dimension_suffix = '3d' if is_3d else '2d'
        final_save_path = f"{base_name}_{dimension_suffix}{ext}"
        
        plt.savefig(final_save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"PCA {dimension_suffix.upper()} 散点图已保存到: {final_save_path}")
    
    plt.close()

def generate_grade_interpretation(df, feature_columns, grades):
    interpretation_data = []
    
    unique_grades = np.unique(grades)
    for grade in unique_grades:
        mask = grades == grade
        grade_samples = df[mask]
        
        stats = {
            'Grade': int(grade),
            'Sample_Count': len(grade_samples),
            'Sample_IDs': ', '.join(grade_samples['Sample_ID'].tolist())
        }
        
        for col in feature_columns:
            if col in grade_samples.columns:
                stats[f'{col}_mean'] = grade_samples[col].mean()
                stats[f'{col}_std'] = grade_samples[col].std()
        
        interpretation_data.append(stats)
    
    interpretation_df = pd.DataFrame(interpretation_data)
    return interpretation_df

def save_grading_results(df_with_grades, original_clusters, output_dir=None):
    if output_dir is None:
        output_dir = TABLES_DIR

    results_df = df_with_grades[['Sample_ID', 'Grade']].copy()
    results_df['Cluster'] = original_clusters 

    results_df = results_df[['Sample_ID', 'Cluster', 'Grade']]
    
    output_path = os.path.join(output_dir, NEW_GRADES_FILE)
    results_df.to_csv(output_path, index=False)
    print(f"分级结果已保存到: {output_path}")
    
    return results_df
