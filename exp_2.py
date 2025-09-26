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

def compute_distances(test_data, train_data):
    """计算测试数据与训练数据间的欧几里得距离"""
    diff = test_data[:, np.newaxis] - train_data
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
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

def cross_validation_loo(data_x, data_y, k_val):
    """执行留一法交叉验证"""
    sample_count = len(data_x)
    correct_count = 0
    
    for i in range(sample_count):
        # 构建训练集和测试集
        test_x = data_x[i:i+1]
        train_x = np.vstack([data_x[:i], data_x[i+1:]])
        train_y = np.hstack([data_y[:i], data_y[i+1:]])
        
        # 计算距离并分类
        dist = compute_distances(test_x, train_x)
        pred = classify_knn(dist, train_y, k_val)[0]
        
        if pred == data_y[i]:
            correct_count += 1
    
    return correct_count / sample_count

def evaluate_test_set(train_x, train_y, test_x, test_y, optimal_k):
    """在测试集上评估模型性能"""
    dist_matrix = compute_distances(test_x, train_x)
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
    
    # 参数设置
    k_candidates = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracy_scores = []
    
    # LOO验证
    for k in k_candidates:
        score = cross_validation_loo(train_features, train_labels, k)
        accuracy_scores.append(score)
    
    # 找到最佳k值
    optimal_idx = np.argmax(accuracy_scores)
    best_k = k_candidates[optimal_idx]
    best_score = accuracy_scores[optimal_idx]
    
    # 输出结果
    print('=' * 60)
    print('训练集LOO验证结果汇总')
    print('=' * 60)
    print('k值\t准确率\t\t百分比')
    print('-' * 40)
    
    for i, (k, acc) in enumerate(zip(k_candidates, accuracy_scores)):
        star = ' ⭐' if i == optimal_idx else ''
        print(f'{k}\t{acc:.4f}\t\t{acc*100:.2f}%{star}')
    
    print('\n' + '=' * 60)
    print('训练集最优结果')
    print('=' * 60)
    print(f'🏆 最佳k值: {best_k}')
    print(f'🎯 最高准确率: {best_score:.4f} ({best_score*100:.2f}%)')
    
    # 测试集验证（可选）
    test_accuracy = evaluate_test_set(train_features, train_labels, test_features, test_labels, best_k)
    print(f'📊 测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')

if __name__ == '__main__':
    main()