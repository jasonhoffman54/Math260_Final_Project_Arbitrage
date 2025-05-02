import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import math


def parseRates(filename):
    """
    Reads the full exchange rate matrix CSV file.
    The CSV file should have currencies as both the row indices (base) and column headers (quote).
    Returns a tuple (currencies, matrix), where:
      - currencies: list of currency codes,
      - matrix: a NumPy array containing exchange rates.
    """
    df = pd.read_csv(filename, index_col=0)
    currencies = df.index.tolist()
    matrix = df.to_numpy().astype(float)
    return currencies, matrix

def buildGraph(rateData):
    """
    Applies the -log transformation to the exchange rate matrix
    to create the graph matrix used for arbitrage detection.
    For each entry: graph[i, j] = -log(rates_matrix[i, j]).
    Assumes that the rates_matrix is complete.
    """
    if np.any(rateData == 0):
        raise ValueError("Exchange rates matrix contains zero(s); cannot take log(0).")
    return -np.log(rateData)

def visualizeGraph(graph, currencies):
    G = nx.DiGraph()
    n = len(currencies)
    for i, currency in enumerate(currencies):
        G.add_node(i, label=currency)
    for i in range(n):
        for j in range(n):
            if i != j:
                weight = graph[i, j]
                if not np.isinf(weight):
                    G.add_edge(i, j, weight=weight)
    
    finite_weights = [data['weight'] for _, _, data in G.edges(data=True) if np.isfinite(data['weight'])]
    min_weight = min(finite_weights) if finite_weights else 0.0
    H = G.copy()
    for _, _, data in H.edges(data=True):
        data['layout_weight'] = data['weight'] - min_weight


    pos = nx.kamada_kawai_layout(H, weight='layout_weight')
    
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
    labels = {i: currencies[i] for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    plt.title("Exchange Rate Graph")
    plt.axis('off')
    plt.show()

def build_edge_list(graph_matrix):
    """
    Converts the graph matrix into a list of edges (i, j, weight) for Bellman-Ford.
    """
    n = graph_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and not np.isinf(graph_matrix[i, j]):
                edges.append((i, j, graph_matrix[i, j]))
    return edges

def bellman_ford(currencies, edges, source):
    """
    Basic Bellman-Ford implementation to detect one negative cycle from a source.
    Returns (dist, pred, negative_cycle_vertex).
    """
    n = len(currencies)
    dist = [float('inf')] * n
    pred = [-1] * n
    dist[source] = 0

    # Relax edges |V|-1 times
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u

    # Check for negative cycle
    negative_cycle = None
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            negative_cycle = v
            pred[v] = u
            break

    return dist, pred, negative_cycle

def reconstruct_negative_cycle(pred, start):
    """
    Reconstructs a negative cycle from pred starting at 'start'.
    """
    n = len(pred)
    curr = start
    # Move into the cycle
    for _ in range(n):
        curr = pred[curr]
    cycle_start = curr
    cycle = [cycle_start]
    curr = pred[cycle_start]
    while curr != cycle_start:
        cycle.append(curr)
        curr = pred[curr]
    cycle.append(cycle_start)
    cycle.reverse()
    return cycle

def compute_cycle_profit(cycle, graph_matrix):
    """
    Computes profit ratio = exp(-sum of weights) for a cycle.
    """
    total_weight = 0.0
    for i in range(len(cycle) - 1):
        u = cycle[i]
        v = cycle[i + 1]
        total_weight += graph_matrix[u, v]
    profit_ratio = math.exp(-total_weight)
    return profit_ratio, total_weight

def bellman_ford_all(currencies, edges, source):
    """
    Extended Bellman-Ford to collect all vertices that can still be relaxed.
    Returns (dist, pred, neg_cycle_vertices).
    """
    n = len(currencies)
    dist = [float('inf')] * n
    pred = [-1] * n
    dist[source] = 0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                updated = True
        if not updated:
            break

    neg_cycle_vertices = []
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            neg_cycle_vertices.append(v)
            pred[v] = u
    return dist, pred, neg_cycle_vertices


def normalize_cycle(cycle):
    """
    Rotate cycle so that the smallest vertex index is first and
    return as a tuple (with last equal to first).
    """
    if cycle[0] == cycle[-1]:
        cycle = cycle[:-1]
    # find rotation point
    min_idx = min(range(len(cycle)), key=lambda i: cycle[i])
    rotated = cycle[min_idx:] + cycle[:min_idx]
    rotated.append(rotated[0])
    return tuple(rotated)

def main():
    filename = "full_exchange_rates_matrix_top20.csv"
    currencies, rates_matrix = parseRates(filename)
    print("Currencies found:", currencies)
    
    graph_matrix = buildGraph(rates_matrix)

    edges = build_edge_list(graph_matrix)
    print(f"Built edge list with {len(edges)} edges for Bellman-Ford.")
    visualizeGraph(graph_matrix, currencies)

if __name__ == "__main__":
    main()