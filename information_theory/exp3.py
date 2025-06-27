import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 字符串MD5 前6位
def md5_6(s):
    # return s
    import hashlib
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    return m.hexdigest()[:6]

# --------------- Huffman Tree Function with Trace --------------- #
def huffman_tree_with_trace(probabilities, symbols, with_len=False):
    len_of_probabilities = len(probabilities)
    len_of_symbols = len(symbols)
    node_counter = 1  # To assign unique IDs

    if len_of_symbols > 2:
        to_add = (len_of_probabilities - 1) % len_of_symbols
        virtual_nodes = np.zeros(len_of_symbols - to_add, dtype=int)
        if to_add > 0:
            probabilities = np.concatenate((probabilities, virtual_nodes))
            len_of_probabilities += len_of_symbols - to_add

    huffman_dict = {}
    graph_edges = []  # For visualization
    
    app = [[]]
    a = []

    for i in range(len(probabilities)):
        key = f"a{i+1}"
        huffman_dict[key] = []
        

    tree_dict = []
    for i in range(len(probabilities)):
        tree_dict.append({"key": f"a{i+1}", "value": probabilities[i]})
        a.append(f"L1:a{i+1}={round(probabilities[i], 2)}")
        
    app.append(a)        
        
    l___ = 1
    while len(tree_dict) > 1:
        l___ += 1
        tree_dict.sort(key=lambda x: x["value"], reverse=True)
        c_prob = 0.0
        collection = []
        node_counter += 1

        a = []
        


        for ind, sym in enumerate(symbols):
            node = tree_dict[-len_of_symbols + ind]
            for symb in node["key"].split("$"):
                huffman_dict[symb].append(sym)
                # graph_edges.append((parent_id, node_map[symb], sym))  # Add directed edge with label
            c_prob += node["value"]
            collection.append(node["key"])
        
        
        for inb in range(len_of_probabilities-len_of_symbols - (l___-2)*2):
            if l___ == len_of_probabilities-len_of_symbols: break
            former = app[l___-1][inb]
            lf = former.split("=")[1]
            k = tree_dict[inb]["key"]
            latter = f"L{l___}:{md5_6(k)}={round(float(lf), 2)}"
            a.append(latter)
            graph_edges.append((former, latter, ""))  # Add directed edge with label
             
        merged_key = "$".join(collection)      
        new_node = f"L{l___}:{md5_6(merged_key)}={round(c_prob, 2)}"
        a.append(new_node)
        a.sort(key=lambda x: float(x.split("=")[1]), reverse=True)
        for inb in range(-len_of_symbols, 0):
            former = app[l___-1][inb]
            graph_edges.append((former, new_node, symbols[inb]))
        
        
        app.append(a)
        
        tree_dict = tree_dict[:-len_of_symbols + 1]

        tree_dict[-1]["key"] = merged_key
        tree_dict[-1]["value"] = c_prob

    huffman_dict.__delitem__(f'a{len_of_probabilities}')
    for i in range(len_of_probabilities - 1):
        rev = list(reversed(huffman_dict[f"a{i+1}"]))
        if with_len:
            huffman_dict[f"a{i+1}"] = ("/".join(rev), len(rev))
        else:
            huffman_dict[f"a{i+1}"] = "/".join(rev)
    return huffman_dict, graph_edges

# --------------- Entropy Function --------------- #
def H(X, symbol_num=2):
    return -np.dot(X, np.log(X) / np.log(symbol_num))

# --------------- Second Order Matrix --------------- #
def MAT_2(X, flatten=False):
    result_ = np.outer(X, X)
    return result_.flatten() if flatten else result_

# --------------- Visualization Functions --------------- #
def draw_huffman_tree(edges, show_details=True):
    G = nx.DiGraph()
    edge_labels = {}
    for parent, child, label in edges:

        
        G.add_edge(parent, child)
        edge_labels[(parent, child)] = label

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=10 if show_details else 4)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10 if show_details else 5)
    plt.title("Huffman Tree")
    plt.show()

def draw_heatmap_with_labels(prob_matrix, huff_codes, title="Huffman Heatmap"):
    T = len(huff_codes)
    n = int(np.sqrt(T))
    if n * n != T:
        matrix = np.array([huff_codes[f"a{i+1}"][0] for i in range(1 * T)]).reshape(1, T)
        values = prob_matrix.reshape(1, T)

        _, ax = plt.subplots(figsize=(16, 7))
        im = ax.imshow(values, cmap='magma', vmin=0, vmax=0.3)

        # Show text in each cell
        for i in range(1):
            for j in range(T):
                text = matrix[i, j]
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
    else:
        matrix = np.array([huff_codes[f"a{i+1}"][0] for i in range(n * n)]).reshape(n, n)
        values = prob_matrix.reshape(n, n)

        _, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(values, cmap='magma', vmin=0, vmax=0.08)

        # Show text in each cell
        for i in range(n):
            for j in range(n):
                text = matrix[i, j]
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)

    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.show()

# ------------------- Main Execution ------------------- #
SYMBOLS = np.array(['-1', '0', '1'])
PROBABILITIES = np.array([0.4, 0.3, 0.1, 0.1, 0.06, 0.04])
huffman_dict_1, edges_1 = huffman_tree_with_trace(PROBABILITIES, SYMBOLS, with_len=True)
draw_huffman_tree(edges_1)
draw_heatmap_with_labels(PROBABILITIES, huffman_dict_1, title="Joint Probability Matrix with Huffman Codes")

PROB2 = MAT_2(PROBABILITIES, flatten=True)
huffman_dict_2, edges_2 = huffman_tree_with_trace(PROB2, SYMBOLS, with_len=True)
draw_huffman_tree(edges_2, show_details=False)

draw_heatmap_with_labels(PROB2, huffman_dict_2, title="Joint Probability Matrix with Huffman Codes")
