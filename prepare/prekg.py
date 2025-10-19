from decimal import Decimal
import json
import ijson
import networkx as nx

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)


# File paths
input_file = '../data/DBLP/dblp-v12-author.json'
output_graph = 'graph/knowledge_graph.graphml'

# Build the knowledge graph
G = nx.MultiDiGraph()

with open(input_file, 'rb') as f:
    for paper in ijson.items(f, 'item'):
        if 'id' in paper:
            paper_id = paper['id']
            G.add_node(paper_id, type='paper', title=paper.get('title', ''), year=paper.get('year', None))

            if 'authors' in paper:
                for author in paper['authors']:
                    auth_id = author.get('id')
                    if auth_id:
                        G.add_node(auth_id, type='author', name=author.get('name', ''))
                        G.add_edge(auth_id, paper_id, relation='writes')

                        org = author.get('org')
                        if org:
                            G.add_node(org, type='organization', name=org)
                            G.add_edge(auth_id, org, relation='affiliated_with')

            if 'venue' in paper:
                venue_id = paper['venue'].get('id')
                if venue_id:
                    venue_raw = paper['venue'].get('raw', '')
                    G.add_node(venue_id, type='venue', name=venue_raw)
                    G.add_edge(paper_id, venue_id, relation='published_in')

            publisher = paper.get('publisher')
            if publisher:
                G.add_node(publisher, type='publisher', name=publisher)
                G.add_edge(paper_id, publisher, relation='published_by')

# Save the graph to GraphML
nx.write_graphml(G, output_graph)

# Print basic stats
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")