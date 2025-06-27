
#2. 接续上面 写出简化的译码表
#3. 从最小汉明距离和H矩阵，分析码字的纠错与检错能力
#4. 如果要构造一个能够纠正1位错误(7， 4) 完备码，写出对应的H矩阵，并进行说明

# 我们继续完成题目中的第 3 和第 4 部分，逐条解析：

# ---

# ## ✅ 第 3 题：最小汉明距离与 H 矩阵分析纠错与检错能力

# ### ▶ 最小汉明距离 $d_{min}$

# 在我们之前构造的码字中：

# ```
# 00000, 01101, 10111, 11010
# ```

# 我们依次计算每对码字之间的汉明距离：

# * $d(00000, 01101) = 3$
# * $d(00000, 10111) = 4$
# * $d(00000, 11010) = 3$
# * $d(01101, 10111) = 3$
# * $d(01101, 11010) = 4$
# * $d(10111, 11010) = 3$

# 因此最小汉明距离为：

# $$
# d_{min} = 3
# $$

# ### ▶ 纠错与检错能力

# 根据线性码的基本理论：

# * 最多可**检错 $d_{min} - 1 = 2$ 位错误**
# * 最多可**纠错 $t = \left\lfloor \frac{d_{min} - 1}{2} \right\rfloor = 1$ 位错误**

# **结论**：该码能**纠正 1 位错误**，**检测 2 位错误**。

# ---

# ## ✅ 对应 H 矩阵（监督矩阵）

# 已知我们是系统线性码，5 位码字，2 位信息位，因此：

# * 总位数：5
# * 信息位：2 → 剩余 3 位为冗余位

# 所以码的参数为 $(n, k) = (5, 2)$，冗余度为 $r = 3$

# 监督矩阵 $H$ 满足：

# $$
# H \cdot c^T = 0
# $$

# 我们可以根据校验公式：

# $$
# \begin{cases}
# c_2 = c_3 + c_4 \\
# c_1 = c_4 \\
# c_0 = c_3 + c_4
# \end{cases}
# \Rightarrow
# c_2 + c_3 + c_4 = 0 \\
# c_1 + c_4 = 0 \\
# c_0 + c_3 + c_4 = 0
# $$

# 每个等式可以写成一行监督向量 $h_i$，组成 $H$：

# $$
# H =
# \begin{bmatrix}
# 0 & 1 & 1 & 1 & 0 \\
# 0 & 0 & 0 & 1 & 1 \\
# 0 & 1 & 0 & 0 & 1 \\
# \end{bmatrix}
# $$

# 列的顺序是：$[c_4, c_3, c_2, c_1, c_0]$

# ---

# ## ✅ 第 4 题：构造一个能纠正1位错误的 (7,4) 完备码（海明码）

# ### ▶ (7,4) Hamming Code 概念

# * **长度 $n = 7$**，信息位 $k = 4$，冗余位 $r = 3$
# * 最小汉明距离 $d_{min} = 3$
# * 可纠正任意 **1 位错误**

# ---

# ### ▶ H 矩阵构造

# 对于 (7,4) Hamming Code，H 矩阵的每一列为 3 位非零二进制向量，共 7 列，且各列两两不相同：

# $$
# H =
# \begin{bmatrix}
# 1 & 0 & 1 & 1 & 1 & 0 & 0 \\
# 1 & 1 & 0 & 1 & 0 & 1 & 0 \\
# 1 & 1 & 1 & 0 & 0 & 0 & 1 \\
# \end{bmatrix}
# $$

# 列号为 $c_1, c_2, c_3, c_4, c_5, c_6, c_7$，每列的编号正是列对应的二进制：

# $$
# \text{列1: }001 \to (1,0,0)^T,\quad \text{列2: }010,\quad \text{列3: }011,\quad \dots
# $$

# ---

# ### ▶ 说明

# 该 $H$ 矩阵：

# * 每一列都是唯一的、非零的 3 位二进制向量
# * 所有单错误对应的 syndromes 唯一
# * 能够检测并定位 1 位错误
# * 对应生成矩阵 $G = [I_4 \mid P]$，可直接用于编码


# 非常好，让我们回到**第三问**，并**分别从最小汉明距离与监督矩阵 $H$** 两个角度来分析该线性码的纠错与检错能力：

# ---

# ## ✅ 第三问：从两个角度分析线性码的纠错与检错能力

# ---

# ### 🧮 1. 从最小汉明距离 $d_{min}$ 出发

# #### ✅ 定义：

# 最小汉明距离是任意两个合法码字之间的最小比特位差异数。

# 从前面已知合法码字：

# ```
# 00000, 01101, 10111, 11010
# ```

# 两两之间汉明距离计算结果为：

# * $d(00000, 01101) = 3$
# * $d(00000, 10111) = 4$
# * $d(00000, 11010) = 3$
# * ...
# * 最小距离为：

#   $$
#   d_{min} = 3
#   $$

# #### ✅ 由此推导：

# * **最大可检测错误数**：

#   $$
#   d_{min} - 1 = 2
#   $$
# * **最大可纠正错误数**：

#   $$
#   t = \left\lfloor \frac{d_{min} - 1}{2} \right\rfloor = 1
#   $$

# #### ✅ 结论（从 $d_{min}$ 角度）：

# 该码**能检测最多 2 位错误，纠正最多 1 位错误**。

# ---

# ### 📐 2. 从监督矩阵 $H$ 出发（伴随式解码）

# 已知监督矩阵：

# $$
# H = \begin{bmatrix}
# 0 & 1 & 1 & 1 & 0 \\
# 0 & 0 & 0 & 1 & 1 \\
# 0 & 1 & 0 & 0 & 1 \\
# \end{bmatrix}
# \quad \text{列顺序：}(c_4, c_3, c_2, c_1, c_0)
# $$

# #### ✅ 性质：

# * $H \cdot c^T = 0$ 对任意合法码字成立（零空间）
# * 接收到的码字 $r$，有：

#   $$
#   s = H \cdot r^T = H(c + e)^T = Hc^T + He^T = He^T
#   $$

#   因为 $Hc^T = 0$

# #### ✅ syndrome 与错误图样一一对应：

# * 若 $e$ 是 1 位错误图样（如 00001 表示最低位错），则每一个 syndrome $s$ 唯一对应一个位置的错误。
# * syndrome 解码表就记录了 $s \mapsto e$

# #### ✅ 结论（从 H 角度）：

# * 监督矩阵 $H$ 的所有列互不线性相关 ⇒ syndrome 能唯一定位**1 位错误**
# * 对应 syndrome 表可以纠正 **1 位错误**
# * 如果错误是 **2 位或以上**，syndrome 会无法唯一对应错误图样 ⇒ **无法纠正，可能误判**

# ---

# ### 📌 总结

# | 分析角度     | 检错能力        | 纠错能力                            | 说明               |
# | -------- | ----------- | ------------------------------- | ---------------- |
# | 最小汉明距离   | $d-1 = 2$ 位 | $\lfloor (d-1)/2 \rfloor = 1$ 位 | 通用定理适用           |
# | 监督矩阵 $H$ | 能区分所有1位错误   | syndrome 唯一 → 纠正1位              | 与 syndrome 解码表一致 |

# 如果你希望我对 syndrome 解码流程做图示或实现实例，我也可以继续扩展。



import itertools
import pandas as pd

# 所有5位二进制字符串
all_words = [''.join(seq) for seq in itertools.product('01', repeat=5)]

# 判断是否在码字中间
def is_valid_codeword(word):
    c4 = int(word[0])
    c3 = int(word[1])
    c2 = int(word[2])
    c1 = int(word[3])
    c0 = int(word[4])
    return (c2 == (c3 ^ c4)) and (c1 == c4) and (c0 == (c3 ^ c4))

# 将所有合法码字放在一个集合当中
valid_codewords = [w for w in all_words if is_valid_codeword(w)]
print("Valid codewords:", valid_codewords)
# 计算码重，为了得到最小汉明距离
def hamming_weight(word):
    return sum(int(b) for b in word)

print("Hamming weights of valid codewords:", [hamming_weight(w) for w in valid_codewords])

# 构造标准阵列
used = set()
standard_array = []

for word in all_words:
    if word in used:
        continue
    row = []
    for codeword in valid_codewords:
        error_vec = '{:05b}'.format(int(word, 2) ^ int(codeword, 2))
        result = '{:05b}'.format(int(codeword, 2) ^ int(word, 2))
        row.append(result)
        used.add(result)
    standard_array.append((word, row))

# 构造成DataFrame输出
df = pd.DataFrame(
    [row for _, row in standard_array],
    index=[leader for leader, _ in standard_array],
    columns=valid_codewords
)

print(df)
import numpy as np

H = np.array([
    [1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1]
])

import numpy as np
# 构造错误图样及其对应的 syndrome（伴随式），并记录汉明重量
error_patterns = []
for bits in itertools.product('01', repeat=5):
    e_str = ''.join(bits)
    e_vec = np.array([int(b) for b in e_str])
    s_vec = H @ e_vec % 2
    s_str = ''.join(map(str, s_vec))
    weight = hamming_weight(e_str)
    error_patterns.append((e_str, s_str, weight))

# 按 syndrome 分组，保留最小汉明重量的错误图样（coset leader）
syndrome_map = {}
for e_str, s_str, w in sorted(error_patterns, key=lambda x: x[2]):
    if s_str not in syndrome_map:
        syndrome_map[s_str] = (e_str, w)

# 整理为 DataFrame 展示
simplified_decode_table = pd.DataFrame(
    [(e, s) for s, (e, _) in sorted(syndrome_map.items())],
    columns=["Error word", "Syndrome"]
)


print(simplified_decode_table)

