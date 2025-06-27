import heapq
from conception import *
# 8. 霍夫曼编码 - 生成码表
def generate_huffman_table(freq_dict):
    """根据频率字典生成霍夫曼码表"""
    heap = [[weight, [symbol, ""]] for symbol, weight in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))

# 9. 霍夫曼编码 - 编码器
def huffman_encode(text, huffman_table):
    """使用霍夫曼码表编码文本"""
    return "".join(huffman_table[char] for char in text)

# 10. 霍夫曼编码 - 解码器
def huffman_decode(encoded_text, huffman_table):
    """使用霍夫曼码表解码文本"""
    reversed_table = {v: k for k, v in huffman_table.items()}
    decoded_text = []
    current_code = ""
    for bit in encoded_text:
        current_code += bit
        if current_code in reversed_table:
            decoded_text.append(reversed_table[current_code])
            current_code = ""
    return "".join(decoded_text)

# 11. 香农-费诺编码 - 生成码表
def shannon_fano_encode(prob_dict):
    """生成香农-费诺编码表"""
    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    codes = {}

    def _recursive_split(sub_list, code):
        if len(sub_list) <= 1:
            if sub_list:
                codes[sub_list[0][0]] = code
            return

        total_prob = sum(item[1] for item in sub_list)
        cumulative_prob = 0
        split_index = -1
        min_diff = float('inf')

        for i, item in enumerate(sub_list):
            cumulative_prob += item[1]
            diff = abs(total_prob - 2 * cumulative_prob)
            if diff < min_diff:
                min_diff = diff
                split_index = i + 1
            else:
                break # 差值开始增大了
        
        _recursive_split(sub_list[:split_index], code + '0')
        _recursive_split(sub_list[split_index:], code + '1')

    _recursive_split(sorted_probs, "")
    return codes


# 12. 计算平均码长
def calculate_avg_code_length(prob_dict, code_table):
    """计算平均码长 L_avg = sum(p_i * l_i)"""
    avg_length = 0
    for symbol, prob in prob_dict.items():
        avg_length += prob * len(code_table[symbol])
    return avg_length

# 13. 计算编码效率
def calculate_coding_efficiency(entropy, avg_length):
    """计算编码效率 η = H(S) / L_avg"""
    if avg_length == 0: return float('inf') # 避免除零
    return entropy / avg_length

# 14. LZW 编码器
def lzw_encode(text):
    """实现LZW压缩算法"""
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w = ""
    result = []
    for c in text:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(dictionary[w])
    return result

# 15. LZW 解码器
def lzw_decode(encoded_sequence):
    """实现LZW解压缩算法"""
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    result = []
    w = dictionary[encoded_sequence.pop(0)]
    result.append(w)
    for k in encoded_sequence:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError("Bad compressed k: %s" % k)
        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        w = entry
    return "".join(result)

# 16. 算术编码 - 编码器 (简化版)
def arithmetic_encode(text, probabilities):
    """对短字符串进行算术编码，返回最终区间"""
    low, high = 0.0, 1.0
    # 预计算每个符号的范围
    ranges = {}
    current_low = 0.0
    for symbol, prob in probabilities.items():
        ranges[symbol] = (current_low, current_low + prob)
        current_low += prob

    for char in text:
        char_low, char_high = ranges[char]
        current_range = high - low
        high = low + current_range * char_high
        low = low + current_range * char_low
        
    return low, high

# 17. 算术编码 - 解码器 (简化版)
def arithmetic_decode(value, probabilities, length):
    """解码由算术编码生成的浮点数"""
    ranges = {}
    current_low = 0.0
    for symbol, prob in probabilities.items():
        ranges[symbol] = (current_low, current_low + prob)
        current_low += prob
        
    decoded_text = ""
    for _ in range(length):
        for symbol, (r_low, r_high) in ranges.items():
            if r_low <= value < r_high:
                decoded_text += symbol
                value = (value - r_low) / (r_high - r_low)
                break
    return decoded_text

# 18. 判断克拉夫特不等式
def check_kraft_inequality(code_lengths, r=2):
    """判断码长列表是否满足克拉夫特不等式"""
    kraft_sum = sum(r**(-l) for l in code_lengths)
    return kraft_sum <= 1

if __name__ == "__main__":
    # --- 示例 ---
    print("--- II. 无损信源编码 ---")
    freqs = {'a': 45, 'b': 13, 'c': 12, 'd': 16, 'e': 9, 'f': 5}
    total_freq = sum(freqs.values())
    probs = {k: v/total_freq for k, v in freqs.items()}

    # 8-10. Huffman
    huff_table = generate_huffman_table(freqs)
    print(f"8. Huffman码表: {huff_table}")
    encoded = huffman_encode('abacaba', huff_table)
    print(f"9. 'abacaba' 编码后: {encoded}")
    decoded = huffman_decode(encoded, huff_table)
    print(f"10. 解码后: {decoded}")

    # 11. Shannon-Fano
    sf_table = shannon_fano_encode(probs)
    print(f"11. Shannon-Fano码表: {sf_table}")

    # 12-13.
    source_entropy = calculate_entropy(probs.values())
    avg_len = calculate_avg_code_length(probs, huff_table)
    print(f"12. Huffman平均码长: {avg_len:.4f}")
    efficiency = calculate_coding_efficiency(source_entropy, avg_len)
    print(f"13. Huffman编码效率: {efficiency:.4f}")

    # 14-15. LZW
    lzw_text = "TOBEORNOTTOBEORTOBEORNOT"
    lzw_encoded = lzw_encode(lzw_text)
    print(f"14. LZW编码 '{lzw_text[:10]}...': {lzw_encoded}")
    lzw_decoded = lzw_decode(lzw_encoded)
    print(f"15. LZW解码后: {lzw_decoded}")

    # 16-17. Arithmetic Coding
    arith_probs = {'A': 0.8, 'B': 0.02, 'C': 0.18}
    arith_text = "ACBA"
    low, high = arithmetic_encode(arith_text, arith_probs)
    print(f"16. 算术编码 '{arith_text}': interval [{low}, {high})")
    value_to_decode = (low + high) / 2
    arith_decoded = arithmetic_decode(value_to_decode, arith_probs, len(arith_text))
    print(f"17. 算术解码 {value_to_decode:.4f}: {arith_decoded}")

    # 18. Kraft
    code_lengths = [len(v) for v in huff_table.values()]
    print(f"18. Huffman码长 {code_lengths} 满足Kraft不等式: {check_kraft_inequality(code_lengths)}\n")