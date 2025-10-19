import json
from collections import defaultdict
import ijson
import networkx as nx

# File paths
input_file = '../data/DBLP/dblp-v12-author.json'
output_graph = 'data/DBLP/citation_network.graphml'

# First pass: Collect all paper ids in the dataset
dense_ids = set()
with open(input_file, 'rb') as f:
    for paper in ijson.items(f, 'item'):
        if 'id' in paper:
            dense_ids.add(paper['id'])

# Second pass: Build the graph
G = nx.DiGraph()
with open(input_file, 'rb') as f:
    for paper in ijson.items(f, 'item'):
        if 'id' in paper:
            pid = paper['id']
            G.add_node(pid)  # Basic node, no extra attributes
            if 'references' in paper:
                for ref in paper['references']:
                    if ref in dense_ids:
                        G.add_edge(pid, ref)  # Directed edge: citing -> cited

# Save to GraphML
nx.write_graphml(G, output_graph)

# Print nodes and edges
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")