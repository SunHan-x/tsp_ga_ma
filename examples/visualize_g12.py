import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path to allow importing tsp_ga_ma
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tsp_ga_ma.data import dist_matrix_g12, vertex_labels_g12

def visualize_graph():
    n = len(dist_matrix_g12)
    G = nx.Graph()

    # Add nodes
    for i in range(n):
        G.add_node(i, label=vertex_labels_g12[i])

    # Add edges
    # Since it's a complete graph, we add all edges.
    # For visualization purposes, we might want to only draw "significant" edges 
    # or just draw them all with transparency.
    # Kamada-Kawai layout needs distances.
    for i in range(n):
        for j in range(i + 1, n):
            dist = dist_matrix_g12[i][j]
            G.add_edge(i, j, weight=dist)

    print("Computing layout...")
    # Kamada-Kawai layout positions nodes using the path-length cost function.
    # It tries to make the geometric distance between nodes proportional to the graph distance.
    pos = nx.kamada_kawai_layout(G, weight='weight')

    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    
    # Draw labels
    labels = {i: vertex_labels_g12[i] for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight='bold')

    # Draw edges
    # We can color edges by distance, or make long edges more transparent
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]
    
    # Normalize weights for colormap (optional) or just draw them simply
    # Let's draw them with low alpha because it's a complete graph
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.2, edge_color='gray')

    # Optionally, highlight the Minimum Spanning Tree to show structure better
    T = nx.minimum_spanning_tree(G, weight='weight')
    nx.draw_networkx_edges(G, pos, edgelist=T.edges(), width=2.0, edge_color='green', alpha=0.8, label='MST')

    # plt.title("Visualization of Graph G_12_66 (Kamada-Kawai Layout)\nGreen edges denote Minimum Spanning Tree")
    plt.axis('off')
    
    output_file = "result/graph_g12_visualization.png"
    os.makedirs("result", exist_ok=True)
    plt.savefig(output_file)
    print(f"Graph visualization saved to {output_file}")
    # plt.show()

if __name__ == "__main__":
    visualize_graph()
