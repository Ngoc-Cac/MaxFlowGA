import networkx as nx, math, copy as cp

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
def give_edge_weights(edges: tuple[tuple[int, int] | tuple[str, str]], weight_matrix: list[list[int]],
                      index_table: dict[str, int] | None = None) -> dict[tuple[int, int]: int]:
    """
    Decode the edge weight of a directed graph.\n\n

    edges: a tuple of edges. Each edge should be a tuple containing two vertices.\n
    weight_matrix: a two-D list containing integers. Each element at (i, j) represents
    the weight of the edge from the ith vertex to the jth vertex.
    """
    if index_table:
        return {edge: weight_matrix[index_table[edge[0]]][index_table[edge[1]]] for edge in edges}
    else:
        return {edge: weight_matrix[edge[0] if edge[0] != 's' else 0][edge[1] if edge[1] != 't' else -1] for edge in edges}
def draw_digraph(graph: nx.DiGraph, nodes_pos: dict[any: tuple[int, int]],
                 edges: list[tuple[any, any]], edge_weights: dict[tuple[any, any]: float | int],*,\
                 node_font_size=12, edge_label_pos=0.3, edge_font_size=10) -> None:
    nx.draw_networkx_nodes(graph, nodes_pos)
    nx.draw_networkx_labels(graph, nodes_pos, font_size=node_font_size)
    nx.draw_networkx_edges(graph,nodes_pos,edges)
    nx.draw_networkx_edge_labels(graph, nodes_pos, edge_weights, label_pos=edge_label_pos, font_size=edge_font_size)