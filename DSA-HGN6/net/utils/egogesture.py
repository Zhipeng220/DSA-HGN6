import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 21  # EgoGesture uses 21 hand joints
self_link = [(i, i) for i in range(num_node)]

# Define hand skeleton connections based on MediaPipe hand model
# Joint indices (0-20):
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index finger (MCP, PIP, DIP, TIP)
# 9-12: Middle finger (MCP, PIP, DIP, TIP)
# 13-16: Ring finger (MCP, PIP, DIP, TIP)
# 17-20: Pinky finger (MCP, PIP, DIP, TIP)

inward_ori_index = [
    # Thumb connections
    (1, 0), (2, 1), (3, 2), (4, 3),
    # Index finger connections
    (5, 0), (6, 5), (7, 6), (8, 7),
    # Middle finger connections
    (9, 0), (10, 9), (11, 10), (12, 11),
    # Ring finger connections
    (13, 0), (14, 13), (15, 14), (16, 15),
    # Pinky finger connections
    (17, 0), (18, 17), (19, 18), (20, 19),
]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    """
    The Graph to model the hand skeleton structure for EgoGesture dataset

    Args:
        labeling_mode: must be one of the follow candidates
            - 'uniform': Uniform Labeling
            - 'distance': Distance Partitioning
            - 'spatial': Spatial Configuration
    """

    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        """
        Get adjacency matrix based on labeling mode
        """
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode '{labeling_mode}' is not supported")
        return A


if __name__ == '__main__':
    # Test the graph
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:10.0'
    A = Graph('spatial').get_adjacency_matrix()

    print("Adjacency matrix shape:", A.shape)
    print("Number of partitions:", A.shape[0])

    for i, a in enumerate(A):
        print(f"\nPartition {i}:")
        plt.imshow(a, cmap='gray')
        plt.title(f'Spatial Configuration - Partition {i}')
        plt.colorbar()
        plt.savefig(f'egogesture_graph_partition_{i}.png')
        plt.clf()

    print("\nGraph structure:")
    print(f"Number of nodes: {num_node}")
    print(f"Number of edges: {len(neighbor)}")
    print(f"Inward edges: {len(inward)}")
    print(f"Outward edges: {len(outward)}")