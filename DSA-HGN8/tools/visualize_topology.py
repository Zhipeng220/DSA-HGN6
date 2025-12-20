# tools/visualize_topology.py (增强版)
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os


def visualize_topology(npy_path, save_name=None, threshold=0.0):
    if save_name is None:
        save_name = npy_path.replace('.npy', '.png')

    print(f"Loading {npy_path}...")
    if not os.path.exists(npy_path):
        print("Error: File not found.")
        return

    A = np.load(npy_path)
    if len(A.shape) == 3: A = A[0]  # (1, V, V) -> (V, V)

    # 简单的归一化，方便可视化权重差异
    # A = (A - A.min()) / (A.max() - A.min() + 1e-9)

    num_nodes = A.shape[0]
    G = nx.Graph()
    for i in range(num_nodes): G.add_node(i)

    # 移除对角线 (自连接)
    np.fill_diagonal(A, 0)

    count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = A[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
                count += 1

    print(f"Plotting {count} edges with threshold > {threshold}...")

    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(G)

    edges = G.edges(data=True)
    # 线宽根据权重动态调整
    weights = [d['weight'] * 2 for (u, v, d) in edges]

    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='#66ccff')
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='#ff5555', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.title(f"Virtual Topology (Epoch X)\n{count} active connections")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"Saved to {save_name}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tools/visualize_topology.py <path_to_npy> [threshold]")
    else:
        path = sys.argv[1]
        thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        visualize_topology(path, threshold=thresh)