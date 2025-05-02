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

def visualizeGraph(currencies, graph_matrix, cycle=None):
    """
    Draws the full exchange‐rate graph and optionally highlights an arbitrage cycle.
    """
    G = nx.DiGraph()
    n = len(currencies)
    for i, curr in enumerate(currencies):
        G.add_node(i, label=curr)
    for u, v, w in build_edge_list(graph_matrix):
        G.add_edge(u, v, weight=w)

    # Shift weights for layout
    finite_ws = [d['weight'] for _,_,d in G.edges(data=True)]
    min_w = min(finite_ws) if finite_ws else 0.0
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d['layout_weight'] = d['weight'] - min_w

    try:
        pos = nx.kamada_kawai_layout(H, weight='layout_weight')
    except Exception as e:
        print("KK layout failed, using spring:", e)
        pos = nx.spring_layout(G, seed=42, k=1.0, iterations=1000)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_size=400, node_color='lightblue',
                           edgecolors='black')
    labels = {i: currencies[i] for i in range(n)}
    nx.draw_networkx_labels(G, pos,
                            labels, font_size=8, font_weight='bold')

    # Highlight cycle edges
    edges_list = list(G.edges())
    cycle_edges = set()
    if cycle:
        cycle_edges = {(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)}
    edge_colors = ['red' if (u,v) in cycle_edges else 'gray'
                   for (u,v) in edges_list]
    widths = [2.5 if (u,v) in cycle_edges else 0.7
              for (u,v) in edges_list]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_list,
        edge_color=edge_colors,
        width=widths,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10
    )

    # Title with profit ratio
    if cycle:
        total_w = sum(graph_matrix[u,v] for u,v in cycle_edges)
        profit = math.exp(-total_w)
        title = f"Best Arbitrage Cycle (×{profit:.4f})"
    else:
        title = "Exchange Rate Graph"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
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


    n = len(currencies)
    best_cycle = None
    best_profit = 1.0001
    cycle_cache = {}

    # Try each currency as source
    for source in range(n):
        _, pred, neg_vertices = bellman_ford_all(currencies, edges, source)
        for v in set(neg_vertices):
            cycle = reconstruct_negative_cycle(pred, v)
            norm = normalize_cycle(cycle)
            if norm in cycle_cache:
                continue
            profit, _ = compute_cycle_profit(cycle, graph_matrix)
            cycle_cache[norm] = profit
            if profit > best_profit:
                best_profit = profit
                best_cycle = cycle

    if best_cycle:
        print("Best arbitrage cycle:", [currencies[i] for i in best_cycle])
        print(f"Profit ratio: {best_profit:.4f}")
        visualizeGraph(currencies, graph_matrix, best_cycle)
    else:
        print("No arbitrage opportunity detected.")
        visualizeGraph(currencies, graph_matrix)

if __name__ == "__main__":
    main()