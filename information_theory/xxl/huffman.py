import numpy as np  # 导入 NumPy 库用于数值运算和数组操作
def huffman_tree(probabilities, symbols, with_len=False):
    # 获取概率列表长度
    len_of_probabilities = len(probabilities)
    # 获取符号集的长度
    len_of_symbols = len(symbols)

    # 如果码符个数大于2（即为多元 Huffman 编码），进行虚拟节点补齐
    if len_of_symbols > 2:
        to_add = (len_of_probabilities - 1) % len_of_symbols  # 计算需补齐的个数
        virtual_nodes = np.zeros(len_of_symbols - to_add, dtype=int)  # 虚拟节点的概率为0
        if to_add > 0:
            # 将虚拟节点拼接到原概率数组中
            probabilities = np.concatenate((probabilities, virtual_nodes))
            len_of_probabilities += len(symbols) - to_add  # 更新概率数组长度

    # 初始化 Huffman 编码字典：每个符号对应一个空编码列表
    huffman_dict = {}
    for i in range(len(probabilities)):
        huffman_dict[f"a{i+1}"] = []

    # 初始化树的结构字典：每个节点包含键名和对应的概率
    tree_dict = []
    for i in range(len(probabilities)):
        tree_dict.append({"key": f"a{i+1}", "value": probabilities[i]})

    # 构建 Huffman 树
    while len(tree_dict) > 1:
        # 将节点按概率从大到小排序（贪心策略）
        tree_dict.sort(key=lambda x: x["value"], reverse=True)

        c_prob = 0.0  # 合并节点后的新概率
        collection = []  # 被合并的 key 集合

        # 遍历当前最低的 len_of_symbols 个节点，进行合并
        for ind, x in enumerate(symbols):
            node = tree_dict[-len_of_symbols + ind]
            # 将合并后的符号加上对应的码符
            for symb in node["key"].split("$"):
                huffman_dict[symb].append(x)
            c_prob += node["value"]
            collection.append(node["key"])

        # 移除已合并的节点，只保留未被合并的 +1 个新节点
        tree_dict = tree_dict[:-len_of_symbols + 1]
        # 更新最后一个节点为新合并节点
        tree_dict[-1]["key"] = "$".join(collection)
        tree_dict[-1]["value"] = c_prob

    # 最终应只剩下一个根节点，且总概率应为 1.0
    assert len(tree_dict) == 1 and tree_dict[0]["value"] == 1.0, "Please CHECK the validation of the input"

    # 删除最后一个虚拟符号（构造中添加的）
    # huffman_dict.__delitem__(f'a{len_of_probabilities}')

    # 对编码结果进行反转并格式化
    for i in range(len_of_probabilities - 1):
        rev = list(reversed(huffman_dict[f"a{i+1}"]))  # 反转编码顺序（编码从根到叶）
        if with_len:
            huffman_dict[f"a{i+1}"] = ("/".join(rev), len(rev))  # 同时返回码长
        else:
            huffman_dict[f"a{i+1}"] = "/".join(rev)  # 只返回编码

    return huffman_dict  # 返回 Huffman 编码结果

def MAT_2(X, flatten=False):
    result_ = []
    for i in range(len(X)):
        result_.append([])
        for j in range(len(X)):
            result_[i].append(X[i] * X[j])  # 构建联合概率矩阵 P(Xi) * P(Xj)

    result_ = np.array(result_)
    if flatten:
        result_ = result_.flatten()  # 将矩阵压平为向量
    return result_


def H(X, symbol_num=2):
    # 计算信息熵，单位为以 symbol_num 为底的对数（默认是二进制熵）
    return - np.dot(X, np.log(X) / np.log(symbol_num))


SYMBOLS = np.array(['0', '1'])  # 定义多元 Huffman 编码符号集（三元）

PROBABILITIES = np.array([0.4, 0.3, 0.1, 0.1, 0.06, 0.04])

# 计算理想平均码长（以三元符号为底），用于评估压缩效率
print("Ideal   ====> ", H(PROBABILITIES)/np.log2(2))

# 构造 Huffman 编码字典，返回编码及其长度
huffman_dict = huffman_tree(PROBABILITIES, SYMBOLS, with_len=True)
print("Huffman  ====> ", huffman_dict)  # 输出 Huffman 编码字典
# 计算平均码长：sum(pi * Li)
mean_n = np.dot(np.array([x[1] for x in list(huffman_dict.values())]), PROBABILITIES)

print(huffman_dict)  # 输出 Huffman 编码字典
print("Mean_n  ====> ", mean_n)  # 输出平均码长（单位为 码符/信源符号）

# 码率计算，一阶情况
rate = H(PROBABILITIES,2) / mean_n  # 码率计算
print("Rate    ====> ", rate)  # 输出码率（单位为 bit/码符）


PROBABILITIES = MAT_2(PROBABILITIES, flatten=True)  # 构造联合概率并展开为一维

# 使用二阶概率重新构造 Huffman 编码
huffman_dict = huffman_tree(PROBABILITIES, SYMBOLS, with_len=True)

# 计算二阶的平均码长（单位为 码符/两个信源符号）
mean_n = np.dot(np.array([x[1] for x in list(huffman_dict.values())]), PROBABILITIES)

# 单位换算成 “码符/单个信源符号”
mean_n = mean_n / 2

print(huffman_dict)  # 输出二阶编码
print("Mean_n  ====> ", mean_n)  # 输出单信源符号的平均码长

# 码率计算，二阶情况，由于是二阶概率，所以需要除以 2
rate = H(PROBABILITIES, 3) / mean_n / 2   # 码率计算
print("Rate    ====> ", rate)  # 输出码率（单位为 bit/码符）