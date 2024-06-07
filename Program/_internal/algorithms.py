from typing import Callable
from Individual import Individual
import random as rand, networkx as nx, math, copy as cp

#Utility functions
#Functions for 2d matrix manipulation
def select_random(population: list[Individual]) -> Individual:
    """Select a random Individual from a pool of Individuals using their fitness as weights\n\n

    population: a list of Individuals to choose from
    """
    return rand.choices(population, [ind.fitness_score for ind in population], k=1)[0]
def col_sum(twoDmatrix: list[list[int]], column: int) -> int:
    """Find the sum of numbers along a column in a two-D matrix\n\n

    twoDmatrix: a two-D list containing numbers
    """
    return sum(row[column] for row in twoDmatrix)

#Ford-Fulkerson Stuff
def augment_path_dfs(res_matrix: list[list[int]]) -> list[int] | None:
    """Finds augmenting path for Ford Fulkerson using DFS\n\n

    res_matrix: a two-D list containing intergers. Each element representing the residual flow
    in network
    """
    visited: list[int] = [False] * len(res_matrix)
    parent_of: list[int] = [-1] * len(res_matrix)
    stack: list[int] = [0] #this is technically a stack
    while stack:
        current = stack.pop()
        visited[current] = True
        for i, value in enumerate(res_matrix[current]):
            if not visited[i] and value:
                stack.append(i)
                parent_of[i] = current
                if i == len(res_matrix) - 1:
                    return parent_of
    return None
def ford_fulkerson(capacity_matrix: list[list[int]]) -> tuple[int, list[list[int]]]:
    """
    An implementation of Ford Fulkerson algorithm to find maximum flow.\n
    This implementation use DFS to find augmenting paths.\n\n

    capacity_matrix: a two-D list containing integers. Each element at (i, j) represents the
    maximum flow from the ith vertex to the jth vertex.
    """
    parent_of: list[int]
    res_matrix: list[list[int]] = cp.deepcopy(capacity_matrix)
    max_flow: int = 0
    cur_flow: int
    node: int
    while (parent_of := augment_path_dfs(res_matrix)):
        cur_flow = math.inf
        node = len(capacity_matrix) - 1
        while node:
            cur_flow = min(cur_flow, res_matrix[parent_of[node]][node])
            node = parent_of[node]
        node = len(capacity_matrix) - 1
        while node:
            res_matrix[parent_of[node]][node] -= cur_flow
            res_matrix[node][parent_of[node]] += cur_flow
            node = parent_of[node]
        max_flow += cur_flow
    return max_flow, res_matrix

#Graph stuff
def fitness(capacity_matrix: list[list[int]], flow_matrix: list[list[int]]) -> float:
    """Calculate fitness of a random flow given its capacity_matrix"""
    temp = Individual()
    temp.dna = flow_matrix
    temp.fitness(min(sum(capacity_matrix[0]), col_sum(capacity_matrix, -1)))
    return temp.fitness_score
def assign_flow(capacity_matrix: list[list[int]], res_matrix: list[list[int]]) -> list[list[int]]:
    """Assign the actual flow given a residual matrix\n\n

    capacity_matrix: a two-D list containing integers. Each element at (i, j) represents the
    maximum flow from the ith vertex to the jth vertex.\n
    res_matrix: a two-D list containing intergers. Each element representing the residual flow
    in network
    """
    flow: list[list[int]] = [[0] * len(capacity_matrix) for _ in range(len(capacity_matrix))]
    for i, row in enumerate(capacity_matrix):
        for j, cap in enumerate(row):
            if cap:
                flow[i][j] = cap - res_matrix[i][j]
    return flow
def generate_edges(adjacent_nodes: dict[int: tuple[int]]) -> tuple[tuple[int, int]]:
    """
    Generate all edges of a graph. Every graph is treated as a directed graph in this implementation\n\n

    adjacent_nodes: a dictionary containing key-value pairs, where key represents a vertex, and the value
    is a tuple containing neighbouring vertices.
    """
    return tuple((node, adjacent) for node, adjacents in adjacent_nodes.items()
                                   for adjacent in adjacents)
def give_edge_weights(edges: tuple[tuple[int, int]], weight_matrix: list[list[int]])\
                    -> dict[tuple[int, int]: int]:
    """
    Decode the edge weight of a directed graph.\n\n

    edges: a tuple of edges. Each edge should be a tuple containing two vertices.\n
    weight_matrix: a two-D list containing integers. Each element at (i, j) represents
    the weight of the edge from the ith vertex to the jth vertex.
    """
    return {edge: weight_matrix[edge[0] if edge[0] != 's' else 0][edge[1] if edge[1] != 't' else -1] for edge in edges}
def draw_digraph(graph: nx.DiGraph, nodes_pos: dict[any: tuple[int, int]],
                 edges: list[tuple[any, any]], edge_weights: dict[tuple[any, any]: float | int],*,\
                 node_font_size=12, edge_label_pos=0.3, edge_font_size=10) -> None:
    nx.draw_networkx_nodes(graph, nodes_pos)
    nx.draw_networkx_labels(graph, nodes_pos, font_size=node_font_size)
    nx.draw_networkx_edges(graph,nodes_pos,edges)
    nx.draw_networkx_edge_labels(graph, nodes_pos, edge_weights, label_pos=edge_label_pos, font_size=edge_font_size)

#Genetic Algorithm
def max_flow_GA(capacity_matrix: list[list[int]], crossover_func: int = 1, *,
                pop_size: int = 500, mutation_rate: float = 0.05,
                max_iter: int = 200, best_max_iter: int | None = 10,
                update_procedure: Callable | None = None) -> dict[str, Individual | int]:
    """Genetic Algorithm on Max Flow Problem\n\n
    
    capacity_matrix: a two-D list of integers. Each element at (i, j) represents the capacity flow\
        from the ith vertex to jth vertex.\n
    crossover_func: the type of crossover function to use. Should be numerals of 1, 2 or 3.\
        '1' use our custom crossover function. '2' use one-point crossover technique. '3' use two-\
        point crossover technique.\n
    pop_size: the population size\n
    mutation_rate: the rate of mutation, should be between 0-1\n
    max_iter: the maximum number of iteration\n
    best_max_iter: the maximum number of iteration that the best individual survives. For example,\
        if best_max_iter = 10, then the algorithm will continue running until no new best individual has\
        been found for 10 consecutive iterations. In case best_max_iter=None, only max_iter is used\
        as a stop condition.\n
    update_procedure: a function that does whatever you want when a new best individual is found.\
        Note: The function needs to have two parameters representing the current iteration and the best\
        individual, When the function is called, the algorithm will pass in these two parameters.\
    """
    if best_max_iter == None:
        best_max_iter = max_iter
    population: list[Individual]
    new_pop: list[Individual] = [Individual(capacity_matrix) for _ in range(pop_size)]
    best_ind: Individual = new_pop[0]
    best_at_gen: int
    iter: int = 0
    best_over_gen: int = 0
    while (iter := iter + 1) < max_iter and (best_over_gen := best_over_gen + 1) < best_max_iter:
        population = new_pop
        new_pop = []
        for ind in population:
            ind.fitness(min(sum(capacity_matrix[0]), col_sum(capacity_matrix, -1)))
            if ind.fitness_score > best_ind.fitness_score:
                best_over_gen = 0
                best_ind = ind
                best_at_gen = iter
                if update_procedure:
                    update_procedure(iter, best_ind)
                    # sleep(0.3)
        while len(new_pop) < pop_size:
            partA = select_random(population)
            partB = select_random(population)
            child = partA.crossover(partB, crossover_func)
            child.mutate(capacity_matrix, mutation_rate)
            new_pop.append(child)

    solution: dict[str, Individual | int] = {"total_gen": iter,
                                             "gen_of_best_ind": best_at_gen,
                                             "best_of_all_gen": best_ind,
                                             "best_in_last_gen": max(population, key=lambda x: x.fitness_score)}
    return solution