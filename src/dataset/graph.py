import networkx as nx
from utils import load_graph


class Connectivity:
    graphs = dict()  # avoid duplicate loading
    paths = dict()
    distances = dict()

    def __init__(self):
        self.graphs = Connectivity.graphs
        self.paths = Connectivity.paths
        self.distances = Connectivity.distances
        # with open('./connectivity/scans.txt') as f:
        #     scans = f.readlines()
        #     for scan in scans:
        #         scan_id = scan.strip()
        #         self.__load__graph(scan_id)

    def __load__graph(self, scan_id):
        G = load_graph(scan_id)
        self.graphs[scan_id] = G
        self.paths[scan_id] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances[scan_id] = dict(nx.all_pairs_dijkstra_path_length(G))

    def get_graph(self, scan_id):
        if scan_id not in self.graphs:
            self.__load__graph(scan_id)
        return self.graphs[scan_id]

    def get_path(self, scan_id, viewpoint_id1, viewpoint_id2):
        if scan_id not in self.paths:
            self.__load__graph(scan_id)
        return self.paths[scan_id][viewpoint_id1][viewpoint_id2]

    def get_distance(self, scan_id, viewpoint_id1, viewpoint_id2):
        if scan_id not in self.distances:
            self.__load__graph(scan_id)
        return self.distances[scan_id][viewpoint_id1][viewpoint_id2]


connectivity = Connectivity()

