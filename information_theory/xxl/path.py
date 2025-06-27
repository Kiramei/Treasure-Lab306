import random
from conception import *

# 19. 二元对称信道(BSC)模拟器
def bsc_simulator(binary_string, p_flip):
    """模拟二进制序列通过BSC信道"""
    received = ""
    for bit in binary_string:
        if random.random() < p_flip:
            received += '1' if bit == '0' else '0'
        else:
            received += bit
    return received

# 20. 计算BSC信道容量
def calculate_bsc_capacity(p_flip):
    """计算BSC信道容量 C = 1 - H(p)"""
    if p_flip == 0 or p_flip == 1:
        return 1.0
    h_p = calculate_entropy([p_flip, 1 - p_flip])
    return 1.0 - h_p

# 21. 重复码 - 编码器
def repetition_encode(binary_string, n):
    """(n, 1)重复码编码器"""
    return "".join([bit * n for bit in binary_string])

# 22. 重复码 - 解码器
def repetition_decode(encoded_string, n):
    """(n, 1)重复码解码器（多数表决）"""
    decoded = ""
    for i in range(0, len(encoded_string), n):
        chunk = encoded_string[i:i+n]
        if chunk.count('1') > chunk.count('0'):
            decoded += '1'
        else:
            decoded += '0'
    return decoded

# 23. 汉明码(7,4) - 编码器
def hamming_7_4_encode(data_bits):
    """为4位数据生成7位汉明码"""
    if len(data_bits) != 4 or not all(c in '01' for c in data_bits):
        raise ValueError("输入必须是4位二进制字符串")
    d = [int(b) for b in data_bits]
    p1 = d[0] ^ d[1] ^ d[3]
    p2 = d[0] ^ d[2] ^ d[3]
    p3 = d[1] ^ d[2] ^ d[3]
    return f"{p1}{p2}{d[0]}{p3}{d[1]}{d[2]}{d[3]}"

# 24. 汉明码(7,4) - 检错与纠错
def hamming_7_4_decode_and_correct(received_word):
    """检测并纠正7位汉明码中的一位错误，并返回原始4位数据"""
    if len(received_word) != 7 or not all(c in '01' for c in received_word):
        raise ValueError("输入必须是7位二进制字符串")
    r = [int(b) for b in received_word]
    
    # 计算伴随式 (Syndrome)
    s1 = r[0] ^ r[2] ^ r[4] ^ r[6]
    s2 = r[1] ^ r[2] ^ r[5] ^ r[6]
    s3 = r[3] ^ r[4] ^ r[5] ^ r[6]
    
    error_pos = s3 * 4 + s2 * 2 + s1 * 1
    
    if error_pos > 0:
        print(f"检测到错误在位置 {error_pos}")
        r[error_pos - 1] ^= 1 # 翻转错误位
    
    # 提取原始数据
    corrected_data = f"{r[2]}{r[4]}{r[5]}{r[6]}"
    return corrected_data

# 25. 奇偶校验码 - 生成
def parity_encode(data, mode='even'):
    """为数据添加奇校验或偶校验位"""
    ones_count = data.count('1')
    if mode == 'even':
        parity_bit = '1' if ones_count % 2 != 0 else '0'
    elif mode == 'odd':
        parity_bit = '0' if ones_count % 2 != 0 else '1'
    else:
        raise ValueError("Mode must be 'even' or 'odd'")
    return data + parity_bit

# 26. 奇偶校验码 - 检测
def parity_detect(data_with_parity, mode='even'):
    """检测带有校验位的数据是否有奇数个错误"""
    ones_count = data_with_parity.count('1')
    if mode == 'even':
        return ones_count % 2 == 0 # 如果1的个数是偶数，则通过
    else: # mode == 'odd'
        return ones_count % 2 != 0 # 如果1的个数是奇数，则通过

if __name__ == "__main__":
    # --- 示例 ---
    print("--- III. 信道与信道编码 ---")
    # 19-20. BSC
    orig_signal = "1011001"
    p_flip = 0.1
    rec_signal = bsc_simulator(orig_signal, p_flip)
    print(f"19. BSC模拟: '{orig_signal}' -> '{rec_signal}' (p={p_flip})")
    print(f"20. BSC信道容量 (p={p_flip}): {calculate_bsc_capacity(p_flip):.4f}")

    # 21-22. Repetition Code
    rep_encoded = repetition_encode("10", 3)
    print(f"21. 重复码编码 '10' (n=3): {rep_encoded}")
    rep_received = "101010" # '111' received as '101', '000' as '010'
    rep_decoded = repetition_decode(rep_received, 3)
    print(f"22. 重复码解码 '{rep_received}': {rep_decoded}")

    # 23-24. Hamming Code
    data = "1011"
    ham_encoded = hamming_7_4_encode(data)
    print(f"23. 汉明(7,4)编码 '{data}': {ham_encoded}")
    ham_received = list(ham_encoded)
    ham_received[4] = '0' if ham_received[4] == '1' else '1' # 引入一位错误
    ham_received = "".join(ham_received)
    print(f"24. 接收到错误码 '{ham_received}'")
    ham_corrected = hamming_7_4_decode_and_correct(ham_received)
    print(f"    纠正后的数据: '{ham_corrected}'")

    # 25-26. Parity Check
    par_data = "1011001"
    par_encoded_even = parity_encode(par_data, 'even')
    print(f"25. 偶校验编码 '{par_data}': {par_encoded_even}")
    print(f"26. 偶校验检测 '{par_encoded_even}': {parity_detect(par_encoded_even, 'even')}")
    par_encoded_even_error = par_encoded_even[:3] + '1' + par_encoded_even[4:]
    print(f"    偶校验检测(有错误) '{par_encoded_even_error}': {parity_detect(par_encoded_even_error, 'even')}\n")