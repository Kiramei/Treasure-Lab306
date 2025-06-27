# 27. 马尔可夫信源的熵率
def calculate_markov_entropy_rate(pi_steady_state, transition_matrix):
    """计算稳态马尔可夫信源的熵率 H_rate = sum(pi_i * H(S_n|S_{n-1}=i))"""
    entropy_rate = 0
    for i, pi_i in enumerate(pi_steady_state):
        h_conditional = calculate_entropy(transition_matrix[i])
        entropy_rate += pi_i * h_conditional
    return entropy_rate

# 28. Burrows-Wheeler 变换 (BWT)
def bwt_transform(text):
    """实现Burrows-Wheeler变换"""
    text += '\x03' # 添加文本结束符 (ETX)
    rotations = [text[i:] + text[:i] for i in range(len(text))]
    rotations.sort()
    bwt_string = "".join(r[-1] for r in rotations)
    original_index = rotations.index(text)
    return bwt_string, original_index

# 29. 逆BWT (Inverse BWT)
def inverse_bwt(bwt_string, original_index):
    """实现逆BWT"""
    n = len(bwt_string)
    table = [""] * n
    for _ in range(n):
        table = sorted([bwt_string[i] + table[i] for i in range(n)])
    return table[original_index].rstrip('\x03')

# 30. 率失真函数计算 (Blahut-Arimoto算法)
# 注意: 这是一个复杂算法的简化实现，用于教学目的。
# 需要 numpy: pip install numpy
def blahut_arimoto(p_x, distortion_matrix, s, max_iter=100, tol=1e-6):
    """
    使用Blahut-Arimoto算法计算给定斜率s下的率失真函数值R(s)和D(s)。
    """
    p_x = np.array(p_x)
    d = np.array(distortion_matrix)
    num_x, num_y = d.shape
    
    # 1. 初始化 q(y|x)
    q_y_given_x = np.ones((num_x, num_y)) / num_y
    
    for _ in range(max_iter):
        # 2a. 计算 q(y)
        q_y = p_x @ q_y_given_x
        
        # 2b. 计算辅助函数 a(x)
        exp_neg_s_d = np.exp(-s * d)
        a_x = (q_y * exp_neg_s_d).sum(axis=1)

        # 2c. 更新 q(y|x)
        new_q_y_given_x = (q_y * exp_neg_s_d) / a_x[:, np.newaxis]
        
        # 检查收敛
        if np.allclose(q_y_given_x, new_q_y_given_x, atol=tol):
            break
        q_y_given_x = new_q_y_given_x
    
    # 3. 计算R(s)和D(s)
    # R = I(X;Y)
    p_xy = p_x[:, np.newaxis] * q_y_given_x
    # 避免 log(0)
    p_xy_safe = p_xy[p_xy > 0]
    q_y_safe = q_y[q_y > 0]
    p_x_safe = p_x[p_x > 0]
    
    h_x = -np.sum(p_x_safe * np.log2(p_x_safe))
    h_y = -np.sum(q_y_safe * np.log2(q_y_safe))
    h_xy = -np.sum(p_xy_safe * np.log2(p_xy_safe))
    R = h_x + h_y - h_xy
    
    # D = E[d(x,y)]
    D = np.sum(p_xy * d)
    
    return R, D

if __name__ == "__main__":
    
    # --- 示例 ---
    print("--- IV. 进阶与应用 ---")
    # 27. Markov
    pi = [0.5, 0.5]
    P = [[0.9, 0.1], [0.5, 0.5]]
    m_rate = calculate_markov_entropy_rate(pi, P)
    print(f"27. 马尔可夫信源熵率: {m_rate:.4f}")

    # 28-29. BWT
    bwt_text = "banana"
    bwt_str, bwt_idx = bwt_transform(bwt_text)
    print(f"28. BWT for '{bwt_text}': ('{bwt_str}', {bwt_idx})")
    inv_bwt_str = inverse_bwt(bwt_str, bwt_idx)
    print(f"29. Inverse BWT: '{inv_bwt_str}'")

    # 30. Rate-Distortion
    p_x = [0.5, 0.5] # 二元信源 P(0)=0.5, P(1)=0.5
    distortion = [[0, 1], [1, 0]] # 汉明失真
    s = 1.0 # 斜率参数
    R_s, D_s = blahut_arimoto(p_x, distortion, s)
    print(f"30. Blahut-Arimoto (s={s}): R(s)={R_s:.4f}, D(s)={D_s:.4f}")