import numpy as np
import time
from collections import Counter

def read_dataset(file_path):
    """读取并处理数据集"""
    data = np.loadtxt(file_path)
    features = data[:, :256]  # 图像特征
    labels_onehot = data[:, 256:]  # 独热编码标签
    labels = np.argmax(labels_onehot, axis=1)  # 转为数字标签
    return features, labels

def euclidean_distance(test_data, train_data):
    """欧几里得距离 - 修复版本"""
    distances = np.zeros((test_data.shape[0], train_data.shape[0]))
    for i in range(test_data.shape[0]):
        diff = test_data[i] - train_data
        distances[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    return distances

def manhattan_distance(test_data, train_data):
    """曼哈顿距离 - 修复版本"""
    distances = np.zeros((test_data.shape[0], train_data.shape[0]))
    for i in range(test_data.shape[0]):
        diff = np.abs(test_data[i] - train_data)
        distances[i] = np.sum(diff, axis=1)
    return distances

def chebyshev_distance(test_data, train_data):
    """切比雪夫距离 - 修复版本"""
    distances = np.zeros((test_data.shape[0], train_data.shape[0]))
    for i in range(test_data.shape[0]):
        diff = np.abs(test_data[i] - train_data)
        distances[i] = np.max(diff, axis=1)
    return distances

def minkowski_distance(test_data, train_data, p=3):
    """闵可夫斯基距离 (p=3) - 修复版本"""
    distances = np.zeros((test_data.shape[0], train_data.shape[0]))
    for i in range(test_data.shape[0]):
        diff = np.abs(test_data[i] - train_data)
        distances[i] = np.power(np.sum(np.power(diff, p), axis=1), 1.0/p)
    return distances

def cosine_distance(test_data, train_data):
    """余弦距离 - 优化版本"""
    eps = 1e-8
    
    # 计算范数
    test_norms = np.linalg.norm(test_data, axis=1, keepdims=True) + eps
    train_norms = np.linalg.norm(train_data, axis=1) + eps
    
    # 计算余弦相似度矩阵
    dot_products = np.dot(test_data, train_data.T)
    cosine_similarities = dot_products / (test_norms * train_norms)
    
    # 转换为距离
    distances = 1 - cosine_similarities
    return distances

def classify_knn(distance_matrix, train_labels, k_neighbors):
    """基于距离矩阵的kNN分类"""
    num_test = distance_matrix.shape[0]
    results = np.zeros(num_test, dtype=int)
    
    for idx in range(num_test):
        # 获取k个最近邻的索引
        neighbor_idx = np.argpartition(distance_matrix[idx], k_neighbors)[:k_neighbors]
        neighbor_labels = train_labels[neighbor_idx]
        # 投票决定类别
        vote_count = Counter(neighbor_labels)
        results[idx] = vote_count.most_common(1)[0][0]
    
    return results

def cross_validation_loo(data_x, data_y, k_val, distance_func):
    """执行留一法交叉验证"""
    sample_count = len(data_x)
    correct_count = 0
    
    for i in range(sample_count):
        # 构建训练集和测试集
        test_x = data_x[i:i+1]
        train_x = np.vstack([data_x[:i], data_x[i+1:]])
        train_y = np.hstack([data_y[:i], data_y[i+1:]])
        
        # 计算距离并分类
        dist = distance_func(test_x, train_x)
        pred = classify_knn(dist, train_y, k_val)[0]
        
        if pred == data_y[i]:
            correct_count += 1
    
    return correct_count / sample_count

def evaluate_test_set(train_x, train_y, test_x, test_y, optimal_k, distance_func):
    """在测试集上评估模型性能"""
    dist_matrix = distance_func(test_x, train_x)
    predictions = classify_knn(dist_matrix, train_y, optimal_k)
    accuracy = np.mean(predictions == test_y)
    return accuracy

def main():
    # 数据加载
    try:
        train_features, train_labels = read_dataset('semeion_train.txt')
        test_features, test_labels = read_dataset('semeion_test.txt')
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")
        return
    
    # 距离方法配置
    distance_methods = {
        '欧几里得距离': euclidean_distance,
        '曼哈顿距离': manhattan_distance,
        '切比雪夫距离': chebyshev_distance,
        '闵可夫斯基距离': minkowski_distance,
        '余弦距离': cosine_distance
    }
    
    # 参数设置
    k_candidates = [1, 3, 5, 7, 9, 11, 13, 15]
    
    # 存储所有结果
    all_results = {}
    best_overall_score = -1
    best_overall_method = None
    best_overall_k = None
    
    # 对每种距离方法进行评估
    for method_name, distance_func in distance_methods.items():
        print(f'\n正在评估 {method_name}...')
        
        accuracy_scores = []
        
        # 对当前距离方法进行k值讨论
        for k in k_candidates:
            score = cross_validation_loo(train_features, train_labels, k, distance_func)
            accuracy_scores.append(score)
        
        # 找到当前距离方法的最佳k值
        optimal_idx = np.argmax(accuracy_scores)
        best_k = k_candidates[optimal_idx]
        best_score = accuracy_scores[optimal_idx]
        
        # 存储当前方法的结果
        all_results[method_name] = {
            'scores': accuracy_scores,
            'best_k': best_k,
            'best_score': best_score
        }
        
        # 更新全局最优结果
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_method = method_name
            best_overall_k = best_k
        
        # 输出当前距离方法的结果表格
        print('=' * 60)
        print(f'{method_name} - 训练集LOO验证结果汇总')
        print('=' * 60)
        print('k值\t准确率\t\t百分比')
        print('-' * 40)
        
        for i, (k, acc) in enumerate(zip(k_candidates, accuracy_scores)):
            star = ' ⭐' if i == optimal_idx else ''
            print(f'{k}\t{acc:.4f}\t\t{acc*100:.2f}%{star}')
        
        print(f'\n{method_name} 最优结果: k={best_k}, 准确率={best_score:.4f} ({best_score*100:.2f}%)')
    
    # 输出全局最优结果
    print('\n' + '=' * 70)
    print('训练集全局最优结果')
    print('=' * 70)
    print(f'🏆 最佳距离方法: {best_overall_method}')
    print(f'🎯 最佳k值: {best_overall_k}')
    print(f'📈 最高准确率: {best_overall_score:.4f} ({best_overall_score*100:.2f}%)')
    
    # 使用最优参数在测试集上验证
    best_distance_func = distance_methods[best_overall_method]
    test_accuracy = evaluate_test_set(train_features, train_labels, test_features, test_labels, 
                                    best_overall_k, best_distance_func)
    print(f'📊 测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')

if __name__ == '__main__':
    main()