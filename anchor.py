import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ijson
import networkx as nx



try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback, no progress bar

def generate_knowledge_graph(file_path: str, target_author_id: str) -> nx.Graph:
    G = nx.Graph()
    with open(file_path, 'r', encoding='utf-8') as f:
        papers = ijson.items(f, 'item')
        for paper in tqdm(papers, desc="Processing papers"):
            authors = paper.get('authors', [])
            author_ids = [str(author.get('id', '')) for author in authors]
            if target_author_id not in author_ids:
                continue

            paper_id = str(paper.get('id', ''))
            G.add_node(paper_id, type='paper', label=paper.get('title', 'Unknown Title'), year=paper.get('year', None))

            for author in authors:
                auth_id = str(author.get('id', ''))
                auth_name = author.get('name', 'Unknown Author')
                auth_org = author.get('org', 'Unknown Org')
                G.add_node(auth_id, type='author', label=auth_name, org=auth_org)
                G.add_edge(auth_id, paper_id, relation='wrote')

            venue = paper.get('venue', {})
            venue_id = str(venue.get('id', ''))
            venue_name = venue.get('raw', 'Unknown Venue')
            if venue_id:
                G.add_node(venue_id, type='venue', label=venue_name)
                G.add_edge(paper_id, venue_id, relation='published_in')

            fos_list = paper.get('fos', [])
            for fos in fos_list:
                fos_name = fos.get('name', '')
                if fos_name:
                    G.add_node(fos_name, type='fos', label=fos_name)
                    G.add_edge(paper_id, fos_name, relation='has_fos', weight=fos.get('w', 0.0))

            references = paper.get('references', [])
            for ref_id in references:
                ref_id_str = str(ref_id)
                G.add_node(ref_id_str, type='paper', label=ref_id_str)
                G.add_edge(paper_id, ref_id_str, relation='cites')

            G.nodes[paper_id]['n_citation'] = paper.get('n_citation', 0)

    if len(G.nodes) == 0:
        print(f"No papers found for author ID '{target_author_id}'.")

    return G

def visualize_graph(G: nx.Graph, output_file: str = 'knowledge_graph.png') -> None:
    if len(G.nodes) > 500:
        print(f"Graph too large ({len(G.nodes)} nodes). Removing 'fos' nodes to simplify.")
        fos_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'fos']
        G.remove_nodes_from(fos_nodes)

    node_colors = []
    for node in G.nodes(data=True):
        node_type = node[1].get('type', 'unknown')
        if node_type == 'author':
            node_colors.append('skyblue')
        elif node_type == 'paper':
            node_colors.append('lightgreen')
        elif node_type == 'venue':
            node_colors.append('orange')
        elif node_type == 'fos':
            node_colors.append('pink')
        else:
            node_colors.append('gray')

    pos = nx.spring_layout(G, seed=42, iterations=20)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_color=node_colors, node_size=500, font_size=8, edge_color='gray')

    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

    plt.title('Knowledge Graph for Author')
    plt.savefig(output_file)
    plt.show()