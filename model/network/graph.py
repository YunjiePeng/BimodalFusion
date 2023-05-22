import numpy as np

class SpatialGraph():
    """ Use skeleton sequences extracted by Openpose/HRNet to construct Spatial-Temporal Graph

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration Partitioning
        - gait_temporal: Gait Temporal Configuration Partitioning
            For more information, please refer to the section 'Partition Strategies' in PGG.
        layout (string): must be one of the follow candidates
        - body_12: Is consists of 12 joints.
            (right shoulder, right elbow, right knee, right hip, left elbow, left knee,
             left shoulder, right wrist, right ankle, left hip, left wrist, left ankle).
            For more information, please refer to the section 'Data Processing' in PGG.
        max_hop (int): the maximal distance between two connected nodes # 1-neighbor
        dilation (int): controls the spacing between the kernel points
    """
    def __init__(self,
                 layout='body_12', # Openpose here represents for body_12
                 strategy='spatial',
                 semantic_level=0,
                 max_hop=1,
                 dilation=1):
        self.layout = layout
        self.strategy = strategy
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node, self.neighbor_link_dic = self.get_layout_info(layout)
        self.num_A = self.get_A_num(strategy)

    def __str__(self):
        return self.A

    def get_A_num(self, strategy):
        if self.strategy == 'uniform':
            return 1
        elif self.strategy == 'distance':
            return 2
        elif (self.strategy == 'spatial') or (self.strategy == 'gait_temporal'):
            return 3
        else:
            raise ValueError("Do Not Exist This Strategy")

    def get_layout_info(self, layout):
        if layout == 'body_12':
            num_node = 12
            neighbor_link_dic = {
                0: [(7, 1), (1, 0), (10, 4), (4, 6),
                     (8, 2), (2, 3), (11, 5), (5, 9),
                     (9, 3), (3, 0), (9, 6), (6, 0)],
                1: [(1, 0), (4, 0), (0, 3), (2, 3), (5, 3)],
                2: [(1, 0), (2, 0)]
            }
            return num_node, neighbor_link_dic
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_edge(self, semantic_level):
        # edge is a list of [child, parent] pairs, regarding the center node as root node
        self_link = [(i, i) for i in range(int(self.num_node / (2 ** semantic_level)))]
        neighbor_link = self.neighbor_link_dic[semantic_level]
        edge = self_link + neighbor_link
        center = []
        if self.layout == 'body_12':
            if semantic_level == 0:
                center = [0, 3, 6, 9]
            elif semantic_level == 1:
                center = [0, 3]
            elif semantic_level == 2:
                center = [0]
        return edge, center

    def get_gait_temporal_partitioning(self, semantic_level):
        if semantic_level == 0:
            if self.layout == 'body_12':
                positive_node = {1, 2, 4, 5, 7, 8, 10, 11}
                negative_node = {0, 3, 6, 9}
        elif semantic_level == 1:
            if self.layout == 'body_12':
                positive_node = {1, 2, 4, 5}
                negative_node = {0, 3}
        elif semantic_level == 2:
            if self.layout == 'body_12':
                positive_node = {1, 2}
                negative_node = {0}
        return positive_node, negative_node
            
    def get_adjacency(self, semantic_level):
        edge, center = self.get_edge(semantic_level)
        num_node = int(self.num_node / (2 ** semantic_level))
        hop_dis = get_hop_distance(num_node, edge, max_hop=self.max_hop)
                
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((num_node, num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1

        normalize_adjacency = normalize_digraph(adjacency)
        # normalize_adjacency = adjacency # withoutNodeNorm

        # normalize_adjacency[a][b] = x
        # when x = 0, node b has no connection with node a within valid hop.
        # when x ≠ 0, the normalized adjacency from node b to node a is x.
        # the value of x is normalized by the number of adjacent neighbor nodes around the node b.

        if self.strategy == 'uniform':
            A = np.zeros((1, num_node, num_node))
            A[0] = normalize_adjacency
            return A
        elif self.strategy == 'distance':
            A = np.zeros((len(valid_hop), num_node, num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
            return A
        elif self.strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((num_node, num_node))
                a_close = np.zeros((num_node, num_node))
                a_further = np.zeros((num_node, num_node))
                for i in range(num_node):
                    for j in range(num_node):
                        if hop_dis[j, i] == hop:
                            j_hop_dis = min([hop_dis[j, _center] for _center in center])
                            i_hop_dis = min([hop_dis[i, _center] for _center in center])
                            if j_hop_dis == i_hop_dis:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif j_hop_dis > i_hop_dis:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            return A
        elif self.strategy == 'gait_temporal':
            A = []
            positive_node, negative_node = self.get_gait_temporal_partitioning(semantic_level)
            for hop in valid_hop:
                a_root = np.zeros((num_node, num_node))
                a_positive = np.zeros((num_node, num_node))
                a_negative = np.zeros((num_node, num_node))
                for i in range(num_node):
                    for j in range(num_node):
                        if hop_dis[j, i] == hop:
                            if i == j:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif j in positive_node:
                                a_positive[j, i] = normalize_adjacency[j, i]
                            else:
                                a_negative[j, i] = normalize_adjacency[j, i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_negative)
                    A.append(a_positive)
            A = np.stack(A)
            return A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    # Calculate the shortest path between nodes
    # i.e. The minimum number of steps needed to walk from one node to another
    A = np.zeros((num_node, num_node)) # Ajacent Matrix
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

test = SpatialGraph()
