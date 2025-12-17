import sys
sys.path.extend(['../'])
from graph import tools

# 21 个关节
num_node = 21
self_link = [(i, i) for i in range(num_node)]

# 定义手部 21 个关节的连接关系 (0-based)
# 0: Wrist
# 1-4: Thumb
# 5-8: Index
# 9-12: Middle
# 13-16: Ring
# 17-20: Pinky
inward_ori_index = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
]
0
inward = inward_ori_index
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
