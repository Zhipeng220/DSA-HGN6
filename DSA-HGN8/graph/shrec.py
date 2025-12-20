import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 22
self_link = [(i, i) for i in range(num_node)]

# SHREC'17 Track Layout (22 Joints)
# 0: Wrist
# 1: Palm
# 2-5: Thumb (Base, Joint1, Joint2, Tip)
# 6-9: Index
# 10-13: Middle
# 14-17: Ring
# 18-21: Pinky
inward_ori_index = [
    # Wrist to Palm
    (0, 1),
    # Palm to Fingers
    (1, 2), (1, 6), (1, 10), (1, 14), (1, 18),
    # Thumb
    (2, 3), (3, 4), (4, 5),
    # Index
    (6, 7), (7, 8), (8, 9),
    # Middle
    (10, 11), (11, 12), (12, 13),
    # Ring
    (14, 15), (15, 16), (16, 17),
    # Pinky
    (18, 19), (19, 20), (20, 21)
]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode '{labeling_mode}' is not supported")
        return A