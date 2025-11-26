import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ijson
import networkx as nx
from collections import defaultdict
import os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback, no progress bar

def generate_temporal_knowledge_graphs(file_path: str, target_author_id: str) -> list:
    graphs_by_year = defaultdict(nx.Graph)
    years = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        papers = ijson.items(f, 'item')
        for paper in tqdm(papers, desc="Processing papers"):
            authors = paper.get('authors', [])
            author_ids = [str(author.get('id', '')) for author in authors]
            if target_author_id not in author_ids:
                continue

            year = paper.get('year', None)
            if year is None:
                continue  # 跳过无年份的论文

            G = graphs_by_year[year]
            years.add(year)

            paper_id = str(paper.get('id', ''))
            G.add_node(paper_id, type='paper', label=paper.get('title', 'Unknown Title'), year=year)

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

    if not graphs_by_year:
        print(f"No papers with years found for author ID '{target_author_id}'.")

    # 返回按年份排序的列表：[(year1, G1), (year2, G2), ...]
    sorted_graphs = sorted(graphs_by_year.items(), key=lambda x: x[0])
    return sorted_graphs

def visualize_temporal_graphs(sorted_graphs: list, output_dir: str = '.') -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for year, G in sorted_graphs:
        if len(G.nodes) > 500:
            print(f"Graph for year {year} too large ({len(G.nodes)} nodes). Removing 'fos' nodes to simplify.")
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

        plt.title(f'Knowledge Graph for Author - Year {year}')
        output_file = os.path.join(output_dir, f'knowledge_graph_{year}.png')
        plt.savefig(output_file)
        plt.close()  # 关闭图形以避免内存问题

