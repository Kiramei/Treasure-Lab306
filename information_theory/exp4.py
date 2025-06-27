import numpy as np

# Pre-defined parameters
H_T = np.array([
    [1,1,1,0,0],
    [1,0,0,1,0],
    [1,1,0,0,1],
])

def _add(args):
    return np.add(*args) % 2
    
def _mul(e1, e2):
    return (e1 @ e2) % 2

def cal_G(H_T):
    _H, _W = H_T.shape
    pre_eyes = np.eye(_W-_H)
    append_fix = H_T[:, :_W-_H].T
    G = np.hstack((pre_eyes, append_fix))
    return G

print("G matrix:")
G = cal_G(H_T)
for row in G:
    for elem in row:
        print(int(elem), end=' ')
    print()

print("=" * 20)

def cal_W_set(G):
    _H, _ = G.shape
    src_set = []
    first_row = [0] * _H
    for x in range(1 << _H):
        row = first_row.copy()
        for i in range(_H):
            if x & (1 << i):
                row[i] = 1
        src_set.append(row)
    src_set = np.array(src_set)
    print("Code words:")
    _res = _mul(src_set, G)
    return _res

W_set = cal_W_set(G)
for word in W_set:
    for elem in word:
        print(int(elem), end='')
    print()
print("=" * 20)

def cal_W_weight(code_words):
    weights = []
    for word in code_words:
        weight = np.sum(word)
        weights.append(weight)
    return np.array(weights)

weights = cal_W_weight(W_set)
print("Weights of code words:")
print('\n'.join([str(int(e)) for e in weights]))
print("=" * 20)

max_hamming_dist = int(np.min(weights[weights > 0]))
print("Maximum Hamming distance:", max_hamming_dist)
print("So, the code can detect", max_hamming_dist - 1, "errors.")
print("And can correct", (max_hamming_dist - 1) // 2, "errors.")
print("=" * 20)

def cal_code_book(code_words):
    _H, _W = code_words.shape
    e_n_set = np.eye(_W, dtype=int)
    code_book = []
    for j in range(_H):
        code_book_col = []
        for i in range(_W):
            e_n = e_n_set[i]
            e_n = _add((e_n, code_words[j])).astype(int).tolist()
            e_ns=''.join([str(e) for e in e_n])
            code_book_col.append(e_ns)
        code_book.append(code_book_col)
    code_book = np.array(code_book).T.tolist()
    cb_1 = code_words.astype(int).tolist()
    idx = [''.join([str(e_) for e_ in e]) for e in cb_1]
    code_book = [idx, *code_book]
    return code_book

cb = cal_code_book(W_set)
print("Code book:")
for code in cb:
    for c in code:
        print(c, end=' ')
    print()
print("=" * 20)


