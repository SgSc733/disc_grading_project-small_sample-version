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
    """
    执行层次聚类
    """
    # 提取特征矩阵
    X = df[feature_columns].values
    
    # 计算距离矩阵
    if method == 'ward' and metric != 'euclidean':
        print(f"警告: Ward方法只支持欧氏距离，将使用euclidean替代{metric}")
        metric = 'euclidean'
    
    # 执行层次聚类
    linkage_matrix = linkage(X, method=method, metric=metric)
    
    return linkage_matrix, X

def plot_dendrogram(linkage_matrix, labels=None, save_path=None, n_clusters=None):
    """
    Plot hierarchical clustering dendrogram with colored clusters
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # 计算颜色阈值
    if n_clusters and n_clusters > 1:
        # 根据聚类数确定颜色阈值
        color_threshold = linkage_matrix[-(n_clusters-1), 2] if n_clusters <= len(linkage_matrix) else None
    else:
        # 默认使用70%的最大距离作为阈值
        color_threshold = 0.7 * max(linkage_matrix[:, 2])
    
    # 绘制树状图
    dendrogram(
        linkage_matrix,
        labels=labels,
        truncate_mode=DENDROGRAM_TRUNCATE_MODE,
        p=DENDROGRAM_P if DENDROGRAM_TRUNCATE_MODE == 'lastp' else 0,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=color_threshold,  # 添加颜色阈值
        above_threshold_color='#BCBCBC'  # 超过阈值的分支使用灰色
    )
    
    # 添加阈值线
    if n_clusters and color_threshold:
        plt.axhline(y=color_threshold, c='red', linestyle='--', linewidth=1, 
                   label=f'Cut for {n_clusters} clusters')
        plt.legend(loc='upper right')
    
    plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
    plt.xlabel('Sample ID', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Dendrogram saved to: {save_path}")
    
    plt.close()

def assign_clusters(linkage_matrix, n_clusters=3):
    """
    根据聚类结果分配簇标签
    """
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    return clusters

def find_optimal_clusters_elbow(linkage_matrix, X, max_clusters=10, plot=True):
    """
    使用肘部法则寻找最佳聚类数
    
    参数:
    - linkage_matrix: 层次聚类链接矩阵
    - X: 原始数据矩阵
    - max_clusters: 最大聚类数
    - plot: 是否绘制肘部图
    
    返回:
    - optimal_k: 建议的最佳聚类数
    """
    from sklearn.metrics import silhouette_score
    
    # 确保不超过样本数
    max_clusters = min(max_clusters, len(X) - 1)
    
    # 计算不同聚类数的指标
    K = range(2, max_clusters + 1)
    inertias = []  # 簇内平方和
    silhouette_scores = []  # 轮廓系数
    
    for k in K:
        # 获取聚类标签
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        
        # 计算簇内平方和（WSS）
        wss = 0
        for i in range(1, k + 1):
            cluster_mask = clusters == i
            if np.sum(cluster_mask) > 0:
                cluster_points = X[cluster_mask]
                cluster_center = np.mean(cluster_points, axis=0)
                wss += np.sum((cluster_points - cluster_center) ** 2)
        inertias.append(wss)
        
        # 计算轮廓系数
        if k < len(X):
            silhouette_avg = silhouette_score(X, clusters)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    # 使用肘部法则找到最佳k值
    if len(inertias) > 2:
        # 计算一阶差分（斜率）
        diff1 = np.diff(inertias)
        # 计算二阶差分（斜率的变化）
        diff2 = np.diff(diff1)
        
        # 找到斜率变化最大的点
        if len(diff2) > 0:
            elbow_idx = np.argmax(np.abs(diff2)) + 1
            optimal_k_elbow = K[elbow_idx]
        else:
            optimal_k_elbow = 5
    else:
        optimal_k_elbow = 5

    
    print(f"\n肘部法则建议: {optimal_k_elbow} 个聚类")
    
    # 绘制肘部图
    if plot and len(inertias) > 0:
        plt.figure(figsize=(10, 6))  # 单图尺寸
        
        # 绘制WSS曲线
        plt.plot(K, inertias, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k_elbow, color='r', linestyle='--', 
                   linewidth=2, label=f'Elbow point: k={optimal_k_elbow}')
        
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Within-cluster Sum of Squares (WSS)', fontsize=12)
        plt.title('Elbow Method for Optimal Cluster Number', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # 标注肘部点
        elbow_idx = K.index(optimal_k_elbow) if optimal_k_elbow in K else 0
        plt.annotate(f'Suggested: k={optimal_k_elbow}', 
                    xy=(optimal_k_elbow, inertias[elbow_idx]),
                    xytext=(optimal_k_elbow + 0.5, inertias[elbow_idx]),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
        
        plt.tight_layout()
        
        # 保存图像
        elbow_plot_path = os.path.join(FIGURES_DIR, 'elbow_analysis.png')
        plt.savefig(elbow_plot_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"\n肘部分析图已保存到: {elbow_plot_path}")
        plt.close()
    
    return optimal_k_elbow

def calculate_cluster_centers(X, clusters):
    """
    计算每个簇的中心点
    """
    unique_clusters = np.unique(clusters)
    centers = {}
    
    for cluster in unique_clusters:
        mask = clusters == cluster
        centers[cluster] = np.mean(X[mask], axis=0)
    
    return centers

def rank_clusters_by_distance(centers, reference_point=None):
    """
    基于到参考点的距离对簇进行排序
    """
    if reference_point is None:
        # 使用原点作为参考点
        reference_point = np.zeros(list(centers.values())[0].shape)
    
    distances = {}
    for cluster, center in centers.items():
        distances[cluster] = np.linalg.norm(center - reference_point)
    
    # 按距离排序（距离越小，等级越高）
    sorted_clusters = sorted(distances.keys(), key=lambda x: distances[x])
    
    # 创建等级映射
    grade_mapping = {}
    for i, cluster in enumerate(sorted_clusters):
        grade_mapping[cluster] = i + 1  # 等级从1开始
    
    return grade_mapping, distances


def plot_pca_scatter(df, cluster_labels, save_path=None):
    """
    Plot PCA scatter plot (supports both 2D and 3D)
    """
    from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting
    
    # Get PCA feature columns
    pca_columns = [col for col in df.columns if col.startswith('PC')]
    
    if len(pca_columns) < 2:
        print("警告: PCA特征不足，无法绘制散点图")
        return
    
    # Determine if 2D or 3D plot
    is_3d = len(pca_columns) >= 3
    
    if is_3d:
        # 3D Plot
        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D scatter plot
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
        
        # Add sample labels
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
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Grade', fontsize=12)
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
    else:
        # 2D Plot
        plt.figure(figsize=FIGURE_SIZE)
        
        # Create 2D scatter plot
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
        
        # Add sample labels
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
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Grade', fontsize=12)
        
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        # Modify filename to indicate 2D or 3D
        base_name = os.path.splitext(save_path)[0]
        ext = os.path.splitext(save_path)[1]
        dimension_suffix = '3d' if is_3d else '2d'
        final_save_path = f"{base_name}_{dimension_suffix}{ext}"
        
        plt.savefig(final_save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"PCA {dimension_suffix.upper()} 散点图已保存到: {final_save_path}")
    
    plt.close()

def generate_grade_interpretation(df, feature_columns, grades):
    """
    生成等级解释表
    """
    interpretation_data = []
    
    unique_grades = np.unique(grades)
    for grade in unique_grades:
        mask = grades == grade
        grade_samples = df[mask]
        
        # 计算该等级的统计信息
        stats = {
            'Grade': int(grade),
            'Sample_Count': len(grade_samples),
            'Sample_IDs': ', '.join(grade_samples['Sample_ID'].tolist())
        }
        
        # 计算每个特征的均值和标准差
        for col in feature_columns:
            if col in grade_samples.columns:
                stats[f'{col}_mean'] = grade_samples[col].mean()
                stats[f'{col}_std'] = grade_samples[col].std()
        
        interpretation_data.append(stats)
    
    interpretation_df = pd.DataFrame(interpretation_data)
    return interpretation_df

def save_grading_results(df, grades, grade_mapping, output_dir=None):
    """
    保存分级结果
    """
    if output_dir is None:
        output_dir = TABLES_DIR
    
    # 创建结果DataFrame
    results_df = df[['Sample_ID', 'Disc_Level']].copy()
    results_df['Cluster'] = grades
    results_df['Grade'] = [grade_mapping[cluster] for cluster in grades]
    
    # 保存结果
    output_path = os.path.join(output_dir, NEW_GRADES_FILE)
    results_df.to_csv(output_path, index=False)
    print(f"分级结果已保存到: {output_path}")
    
    return results_df