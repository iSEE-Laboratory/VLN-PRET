import json
import numpy as np
import networkx as nx


class DTW:
  """Dynamic Time Warping (DTW) evaluation metrics.
  Python doctest:
  >>> graph = nx.grid_graph([3, 4])
  >>> prediction = [(0, 0), (1, 0), (2, 0), (3, 0)]
  >>> reference = [(0, 0), (1, 0), (2, 1), (3, 2)]
  >>> dtw = DTW(graph)
  """
  def __init__(self, graph, weight='weight', threshold=3.0):
    """Initializes a DTW object.
    Args:
        graph: networkx graph for the environment.
        weight: networkx edge weight key (str).
        threshold: distance threshold $d_{th}$ (float).
    """
    self.graph = graph
    self.weight = weight
    self.threshold = threshold
    self.distance = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight=self.weight))

  def __call__(self, prediction, reference, metric='sdtw'):
    """Computes DTW metrics.
    Args:
        prediction: list of nodes (str), path predicted by agent.
        reference: list of nodes (str), the ground truth path.
        metric: one of ['ndtw', 'sdtw', 'dtw'].
    Returns:
        the DTW between the prediction and reference path (float).
    """
    assert metric in ['ndtw', 'sdtw', 'dtw']

    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
        for j in range(1, len(reference)+1):
            best_previous_cost = min(dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
            cost = self.distance[prediction[i-1]][reference[j-1]]
            dtw_matrix[i][j] = cost + best_previous_cost
    dtw = dtw_matrix[len(prediction)][len(reference)]

    if metric == 'dtw':
        return dtw

    ndtw = np.exp(-dtw / (self.threshold * len(reference)))
    if metric == 'ndtw':
        return ndtw

    success = self.distance[prediction[-1]][reference[-1]] <= self.threshold
    return success * ndtw


def ndtw_initialize():
    ndtw_criterion = {}
    with open('./data/R2R/id_paths.json') as f:
        scan_gts = json.load(f)
    
    all_scan_ids = []
    for key in scan_gts:
        path_scan_id = scan_gts[key][0]
        if path_scan_id not in all_scan_ids:
            all_scan_ids.append(path_scan_id)
            ndtw_graph = load_graph(path_scan_id)
            ndtw_criterion[path_scan_id] = DTW(ndtw_graph)
    return ndtw_criterion


import json
import networkx as nx
from config import CONNECTIVITY_DIR
def load_graph(scan):
    """Loads a networkx graph for a given scan.
    Args:
        connections_file: A string with the path to the .json file with the
            connectivity information.
    Returns:
        A networkx graph.
    """
    connections_file = f'{CONNECTIVITY_DIR}/{scan}_connectivity.json'
    with open(connections_file) as f:
        lines = json.load(f)
        nodes = np.array([x['image_id'] for x in lines])
        matrix = np.array([x['unobstructed'] for x in lines])
        mask = np.array([x['included'] for x in lines])

        matrix = matrix[mask][:, mask]
        nodes = nodes[mask]

        # pos2d = {x['image_id']: np.array(x['pose'])[[3, 7]] for x in lines}
        pos3d = {x['image_id']: np.array(x['pose'])[[3, 7, 11]] for x in lines}

    graph = nx.from_numpy_array(matrix)
    graph = nx.relabel.relabel_nodes(graph, dict(enumerate(nodes)))

    # nx.set_node_attributes(graph, pos2d, 'pos2d')
    nx.set_node_attributes(graph, pos3d, 'pos3d')

    # weight2d = {(u, v): np.linalg.norm(pos2d[u] - pos2d[v]) for u, v in graph.edges}
    weight3d = {(u, v): np.linalg.norm(pos3d[u] - pos3d[v]) for u, v in graph.edges}
    # nx.set_edge_attributes(graph, weight2d, 'weight2d')
    nx.set_edge_attributes(graph, weight3d, 'weight')

    return graph
