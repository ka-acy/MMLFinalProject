import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_node("Video Frames")
G.add_node("Captions")
G.add_node("Vision Encoder")
G.add_node("Text Encoder")
G.add_node("Feature Fusion")
G.add_node("Classification Head")
G.add_node("Output: Event Labels")

# Add edges
G.add_edges_from([
    ("Video Frames", "Vision Encoder"),
    ("Captions", "Text Encoder"),
    ("Vision Encoder", "Feature Fusion"),
    ("Text Encoder", "Feature Fusion"),
    ("Feature Fusion", "Classification Head"),
    ("Classification Head", "Output: Event Labels")
])

# Draw the graph
pos = nx.spring_layout(G)  # Positioning algorithm
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=10, font_weight="bold", arrows=True)
plt.title("High-Level Model Architecture")
plt.show()
