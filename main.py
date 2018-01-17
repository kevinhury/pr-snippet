"""
stam
"""

from collections import defaultdict
import operator
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from graph import Graph


def iterative_page_rank(graph: Graph, visit=None, epsilon=-1.0, max_iterations=100):
    """
    Iterative PageRank algorithm:
    PR(A) = (1 - d) / N + d * (PR(Ti) / C(Ti) + ... + PR(Tn) / C(Tn))
    
    - PR(A) is the PageRank of Page A
    - PR(Ti) is the PageRank of pages Ti which links to page A
    - C(Ti) is the number of outbound links on page Ti
    - d is a damping factor which can be set between 0 and 1 (usually 0.85)
    """
    damping_factor = 0.85
    reversed_graph = graph.reversed_graph()
    verticies = graph.get_verticies()
    num_verticies = graph.get_num_verticies()
    prev_rank = {vertex: (1 / num_verticies) for vertex in verticies}
    curr_rank = {vertex: 1.0 for vertex in verticies}
    for iter_index in range(max_iterations):
        for v in verticies:
            sigma = 0
            for degin_v in reversed_graph.get_neighbors(v):
                sigma += prev_rank[degin_v] / len(graph.get_neighbors(degin_v))
            if sigma == 0:
                sigma = 1.0
            curr_rank[v] = (1.0 - damping_factor) / num_verticies + damping_factor * sigma
        prev_rank_list = list(prev_rank.values())
        prev_magnitude = np.sqrt(np.dot(prev_rank_list, prev_rank_list))
        prev_rank = curr_rank
        curr_rank_list = list(curr_rank.values())
        magnitude = np.sqrt(np.dot(curr_rank_list, curr_rank_list))
        curr_rank = {vertex: 1.0 for vertex in verticies}
        if visit:
            visit(iter_index, prev_rank)
        if abs(prev_magnitude - magnitude) < epsilon:
            break
        

    ranks = [(v, r) for (v, r) in prev_rank.items() if len(reversed_graph.get_neighbors(v)) > 0]
    return sorted(ranks, reverse=True, key=operator.itemgetter(1))


def matrix_page_rank(graph: Graph, visit=None, epsilon=-1.0, max_iterations=100):
    """
    Matrix PageRank algorithm
    """
    def create_rank_matrix(graph: Graph, matrix_dict: dict):
        """
        Create transpose rank matrix
        """
        verticies = graph.get_verticies()
        num_verticies = graph.get_num_verticies()
        matrix_rank = np.zeros((num_verticies, num_verticies))
        for vertex, index in matrix_dict.items():
            neighbors = graph.get_neighbors(vertex)
            if len(neighbors) > 0:
                rank = 1 / len(neighbors)
            else:
                rank = 0
            for neighbor in neighbors:
                matrix_rank[index, matrix_dict[neighbor]] = rank
        t = np.transpose(matrix_rank)
        zero_cols = (t == 0).all(axis=0)
        t[:, zero_cols] = (1 / graph.get_num_verticies())
        return t

    verticies = graph.get_verticies()
    initial_rank = 1 / len(verticies)
    matrix_dict = {vertex: index for index, vertex in enumerate(verticies)}
    matrix = create_rank_matrix(graph, matrix_dict)
    reversed_graph = graph.reversed_graph()
    curr_rank = np.zeros(graph.get_num_verticies())
    curr_rank.fill(initial_rank)
    for index in range(max_iterations):
        prev_magnitude = np.sqrt(np.dot(curr_rank, curr_rank))
        curr_rank = np.dot(matrix, curr_rank)
        magnitude = np.sqrt(np.dot(curr_rank, curr_rank))
        if visit:
            rank = {k: v for k, v in zip(matrix_dict, curr_rank)}
            visit(index, rank)
        if abs(prev_magnitude - magnitude) < epsilon:
            break

    rank_list = list()
    for index, rank in enumerate(curr_rank):
        rank_list.append((verticies[index], rank))

    rank_list = [(v, r) for (v, r) in rank_list if len(reversed_graph.get_neighbors(v)) > 0]
    return sorted(rank_list, reverse=True, key=operator.itemgetter(1))


def run_interative(graph: Graph, max_iterations=100, epsilon=-1.0, plot_func=None):
    max_iterations = 100
    verticies = graph.get_verticies()
    num_verticies = graph.get_num_verticies()
    plot_mtrx = np.zeros((max_iterations, num_verticies))
    num_iterations = 0

    def iter_iterative(index, rank):
        nonlocal num_iterations
        plot_mtrx[index, :] = list(rank.values())
        num_iterations = index + 1

    ranks = iterative_page_rank(graph, visit=iter_iterative, epsilon=epsilon, max_iterations=max_iterations)

    print("Iterative Page Rank:")
    rank_sum = np.sum(list(map(lambda x: x[1], ranks)))
    for rank in ranks:
        print(rank)
    print("sums to: {}\n".format(rank_sum))

    if plot_func:
        plot_mtrx = plot_mtrx[:num_iterations,:]
        plot_func(plot_mtrx, verticies, num_iterations)


def run_matrix(graph: Graph, max_iterations=100, epsilon=-1.0, plot_func=None):
    verticies = graph.get_verticies()
    num_verticies = graph.get_num_verticies()
    plot_mtrx = np.zeros((max_iterations, num_verticies))
    num_iterations = 0

    def iter_matrix(index, rank):
        nonlocal num_iterations
        plot_mtrx[index, :] = list(rank.values())
        num_iterations = index + 1

    ranks = matrix_page_rank(graph, visit=iter_matrix, epsilon=epsilon, max_iterations=max_iterations)

    # print("Matrix Page Rank:")
    rank_sum = np.sum(list(map(lambda x: x[1], ranks)))
    for rank in ranks:
        print(rank)
    print("sums to: {}\n".format(rank_sum))

    if plot_func:
        plot_mtrx = plot_mtrx[:num_iterations,:]
        plot_func(plot_mtrx, verticies, num_iterations)


def main():
    def plot_func(plot_mtrx, verticies, num_iterations):
        num_verticies = len(verticies)

        fig = plt.figure(figsize=(12, 8))
        for v_index in range(num_verticies):
            sqrt_verticies = int(math.sqrt(num_verticies))
            v_rank = plot_mtrx[: ,v_index]
            x = list(range(1, num_iterations + 1))
            y = plot_mtrx[:, v_index]
            plt.subplot(sqrt_verticies, math.ceil(num_verticies / sqrt_verticies), v_index + 1)
            plt.ylim([np.min(plot_mtrx) * 0.9, np.max(plot_mtrx) * 1.1])
            plt.xlabel('Iteration')
            plt.ylabel('Page Rank')
            plt.plot(x, y, '-b')
            plt.title("Rank of Page #{}".format(verticies[v_index]))
            plt.grid()
        fig.tight_layout()
        plt.show()

        mag = np.apply_along_axis(np.linalg.norm, 1, plot_mtrx)
        convergence = np.ediff1d(mag)
        plt.title("Cost Function")
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.plot(list(range(1, num_iterations)), np.abs(convergence))
        plt.show()

    graph = load_json_graph('graph1.json')

    run_interative(graph, epsilon=0.00001, max_iterations=100, plot_func=plot_func)
    run_matrix(graph, epsilon=0.00001, max_iterations=100, plot_func=plot_func)


def load_json_graph(graphName: str):
    graph = Graph()
    json_graph = json.load(open(graphName))
    graph.add_edges(json_graph)
    return graph


if __name__ == "__main__":
    main()
