import math
import collections
import numpy as np

# 1. 计算香农熵 (Shannon Entropy)
def calculate_entropy(probabilities):
    """计算给定概率分布的香non熵 H(X) = -sum(p * log2(p))"""
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# 2. 计算联合熵 (Joint Entropy)
def calculate_joint_entropy(joint_prob_matrix):
    """计算联合概率分布矩阵的联合熵 H(X, Y)"""
    joint_entropy = 0
    for row in joint_prob_matrix:
        for p in row:
            if p > 0:
                joint_entropy -= p * math.log2(p)
    return joint_entropy

# 3. 计算条件熵 (Conditional Entropy)
def calculate_conditional_entropy(joint_prob_matrix):
    """计算条件熵 H(Y|X) = H(X,Y) - H(X)"""
    # 确保是numpy数组以便于操作
    joint_prob_matrix = np.array(joint_prob_matrix)
    
    # 计算 H(X, Y)
    h_xy = calculate_joint_entropy(joint_prob_matrix)
    
    # 计算X的边缘概率分布 P(x)
    p_x = np.sum(joint_prob_matrix, axis=1)
    
    # 计算 H(X)
    h_x = calculate_entropy(p_x)
    
    # H(Y|X) = H(X,Y) - H(X)
    return h_xy - h_x

# 4. 计算互信息 (Mutual Information)
def calculate_mutual_information(joint_prob_matrix):
    """计算互信息 I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    joint_prob_matrix = np.array(joint_prob_matrix)
    
    p_x = np.sum(joint_prob_matrix, axis=1) # P(X)
    p_y = np.sum(joint_prob_matrix, axis=0) # P(Y)
    
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_xy = calculate_joint_entropy(joint_prob_matrix)
    
    return h_x + h_y - h_xy

# 5. 计算KL散度 (Kullback-Leibler Divergence)
def calculate_kl_divergence(p_dist, q_dist):
    """计算KL散度 D_KL(P || Q)"""
    divergence = 0
    for p, q in zip(p_dist, q_dist):
        if p > 0 and q > 0:
            divergence += p * math.log2(p / q)
    return divergence

# 6. 计算交叉熵 (Cross-Entropy)
def calculate_cross_entropy(p_true, q_pred):
    """计算交叉熵 H(P, Q) = -sum(p * log2(q))"""
    cross_entropy = 0
    for p, q in zip(p_true, q_pred):
        if p > 0 and q > 0:
            cross_entropy -= p * math.log2(q)
    return cross_entropy

# 7. 基于文本计算信源熵
def calculate_text_entropy(text):
    """根据文本中字符频率估算信源熵"""
    if not text:
        return 0
    counter = collections.Counter(text)
    total_chars = len(text)
    probabilities = [count / total_chars for count in counter.values()]
    return calculate_entropy(probabilities)

if __name__ == "__main__":
    # --- 示例 ---
    print("--- I. 基础概念计算 ---")
    probs1 = [0.5, 0.25, 0.25]
    print(f"1. P={probs1} 的熵: {calculate_entropy(probs1):.4f}")

    joint_probs = [[0.25, 0.25], [0.5, 0]]
    print(f"2. 联合概率 P(X,Y)={joint_probs} 的联合熵: {calculate_joint_entropy(joint_probs):.4f}")
    print(f"3. 条件熵 H(Y|X): {calculate_conditional_entropy(joint_probs):.4f}")
    print(f"4. 互信息 I(X;Y): {calculate_mutual_information(joint_probs):.4f}")

    p_dist = [0.1, 0.4, 0.5]
    q_dist = [0.2, 0.3, 0.5]
    print(f"5. P={p_dist}, Q={q_dist} 的KL散度: {calculate_kl_divergence(p_dist, q_dist):.4f}")
    print(f"6. P={p_dist}, Q={q_dist} 的交叉熵: {calculate_cross_entropy(p_dist, q_dist):.4f}")

    text1 = "hello world"
    print(f"7. 文本 '{text1}' 的熵: {calculate_text_entropy(text1):.4f}\n")