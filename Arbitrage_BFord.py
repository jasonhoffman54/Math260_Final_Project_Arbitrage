import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


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



def main():
    filename = "full_exchange_rates_matrix_top20.csv"
    currencies, rates_matrix = parseRates(filename)
    print("Currencies found:", currencies)
    
    graph_matrix = buildGraph(rates_matrix)
    visualizeGraph(graph_matrix, currencies)

if __name__ == "__main__":
    main()