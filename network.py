import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import os

def build_similarity_graph(embeddings, filenames, threshold=0.9):
    """
    Create a similarity graph based on cosine similarity.
    
    :param embeddings: Numpy array of shape (num_images, feature_dim)
    :param filenames: List of filenames corresponding to embeddings
    :param threshold: Similarity threshold for creating edges
    :return: NetworkX graph
    """
    G = nx.Graph()
    num_images = len(filenames)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Add nodes
    for i in range(num_images):
        G.add_node(filenames[i])

    # Add edges if similarity is above threshold
    for i in range(num_images):
        for j in range(i + 1, num_images):  # Avoid duplicate calculations
            if similarity_matrix[i, j] >= threshold:
                G.add_edge(filenames[i], filenames[j], weight=similarity_matrix[i, j])

    return G


def build_similarity_graph(embeddings, filenames, threshold=0.9):
    """
    Create a similarity graph based on cosine similarity.
    
    :param embeddings: Numpy array of shape (num_images, feature_dim)
    :param filenames: List of filenames corresponding to embeddings
    :param threshold: Similarity threshold for creating edges
    :return: NetworkX graph
    """
    G = nx.Graph()
    num_images = len(filenames)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Add nodes
    for filename in filenames:
        G.add_node(filename)

    # Add edges if similarity is above threshold
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):  # Avoid duplicate calculations
            if similarity_matrix[i, j] >= threshold:
                G.add_edge(filenames[i], filenames[j], weight=similarity_matrix[i, j])

    return G


def plot_interactive_graph(G, title="Interactive Coin Similarity Graph"):
    """
    Creates an interactive visualization of the similarity graph using Plotly.
    
    :param G: NetworkX graph
    :param title: Title for the graph
    :return: Plotly figure
    """
    pos = nx.kamada_kawai_layout(G)  # pos = nx.spring_layout(G, seed=42)  # Force-directed layout
    edge_x, edge_y = [], []
    
    # Create edges
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="gray"),
        mode="lines",
        hoverinfo="none"
    )

    # Create nodes
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)  # Show filename on hover

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(size=10, color="blue")
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig


def save_graph(fig, filename_base="coin_graph"):
    """
    Saves the interactive graph in multiple formats.
    
    :param fig: Plotly figure
    :param filename_base: Base name for the saved files
    """
    output_dir = "graphs_output"
    os.makedirs(output_dir, exist_ok=True)

    # Save as HTML for GitHub
    html_path = os.path.join(output_dir, f"{filename_base}.html")
    fig.write_html(html_path)
    print(f"Saved interactive graph to: {html_path}")

    # Save as PNG for LaTeX
    png_path = os.path.join(output_dir, f"{filename_base}.png")
    # fig.write_image(png_path)
    print(f"Saved static image to: {png_path}")


def generate_graphs(embedding_files, filename_files, graph_types):
    """
    Generates similarity graphs with different thresholds for different embedding types.
    
    :param embedding_files: Dictionary with embedding types as keys and file paths as values
    :param filename_files: Dictionary with filename types as keys and file paths as values
    :param graph_types: List of types (e.g., ["obverse", "reverse"])
    """
    for graph_type in graph_types:
        embeddings = np.load(embedding_files[graph_type])
        filenames = np.loadtxt(filename_files[graph_type], dtype=str, usecols=0)

        # Decide the threshold dynamically
        if "fused" in graph_type:
            thresholds = [0.9995, 0.9997, 0.9999]  # Fused embeddings
        else:
            thresholds = [0.9, 0.93, 0.95]  # Image-only embeddings

        for threshold in thresholds:
            print(f"Generating {graph_type} graph with threshold {threshold}...")

            # Build, visualize, and save
            G = build_similarity_graph(embeddings, filenames, threshold)
            fig = plot_interactive_graph(G, title=f"{graph_type.capitalize()} Similarity Graph (Threshold={threshold})")
            save_graph(fig, filename_base=f"{graph_type}_graph_{threshold}")


if __name__ == "__main__":
    embedding_files = {
    "obverse_fused": "fused_embeddings_ob.npy",
    "reverse_fused": "fused_embeddings_rev.npy",
    "obverse_img": "obverse_features_gray.npy",
    "reverse_img": "reverse_features_gray.npy"
    }
    filename_files = {
        "obverse_fused": "fused_filenames_ob.txt",
        "reverse_fused": "fused_filenames_rev.txt",
        "obverse_img": "obverse_filenames_gray.txt",
        "reverse_img": "reverse_filenames_gray.txt"
    }
    graph_types = ["obverse_fused", "reverse_fused", "obverse_img", "reverse_img"]

    generate_graphs(embedding_files, filename_files, graph_types)