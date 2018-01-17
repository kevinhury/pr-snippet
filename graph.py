"""
Graph
"""

from collections import defaultdict

class Graph:
    """
    graph implementation
    """
    def __init__(self):
        self._graph = defaultdict(list)

    def add_edge(self, vertex_u, vertex_v):
        """
        Adding edge to graph
        """
        self._graph[vertex_u].append(vertex_v)
        if self._graph.get(vertex_v) is None:
            self._graph[vertex_v] = list()

    def add_edges(self, graph_dict):
        """
        Adding edges to graph from dict
        """
        for v, us in graph_dict.items():
            for u in us:
                self.add_edge(v, u)

    def reversed_graph(self):
        """
        Returns a new reversed graph.
        """
        reverse = Graph()
        for vertex, neighbors in self._graph.items():
            for neighbor in neighbors:
                reverse.add_edge(neighbor, vertex)
        return reverse

    def print_graph(self):
        """
        Print graph
        """
        print(self._graph)

    def get_verticies(self):
        """
        Returns list of verticies.
        """
        return list(self._graph.keys())

    def get_num_verticies(self):
        """
        Returns num of veritices.
        """
        return len(self.get_verticies())

    def get_neighbors(self, vertex):
        """
        Returns list of neighbors of given vertex.
        """
        return self._graph.get(vertex)
