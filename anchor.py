import json
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')  # 或者试试 'QtAgg' 如果TkAgg不行
import matplotlib.pyplot as plt
from typing import List, Dict, Any

import json
import ijson  # 需要安装: pip install ijson
from tqdm import tqdm  # 进度条，假设你的环境有
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any


def generate_knowledge_graph(file_path: str, target_author_id: str) -> nx.Graph:
    """
    使用 ijson 流式读取大JSON文件，只处理目标作者的论文，生成知识图谱。
    显著降低内存使用和时间。
    """
    G = nx.Graph()

    with open(file_path, 'r', encoding='utf-8') as f:
        # ijson 逐项解析JSON数组
        papers = ijson.items(f, 'item')  # 假设JSON是数组，'item' 表示每个元素

        # 使用 tqdm 监控进度（假设文件有约数百万项，你可以估算total）
        for paper in tqdm(papers, desc="Processing papers", total=1000000):  # total是估算值，调整为你的文件大小
            authors = paper.get('authors', [])
            author_ids = [author.get('id', '') for author in authors]

            if target_author_id not in author_ids:
                continue  # 跳过无关论文，节省时间

            # 以下同原代码，添加节点和边
            paper_id: str = paper.get('id', '')
            G.add_node(paper_id, type='paper', label=paper.get('title', 'Unknown Title'), year=paper.get('year', None))

            for author in authors:
                auth_id: str = author.get('id', '')
                auth_name: str = author.get('name', 'Unknown Author')
                auth_org: str = author.get('org', 'Unknown Org')
                G.add_node(auth_id, type='author', label=auth_name, org=auth_org)
                G.add_edge(auth_id, paper_id, relation='wrote')

            venue = paper.get('venue', {})
            venue_id: str = venue.get('id', '')
            venue_name: str = venue.get('raw', 'Unknown Venue')
            if venue_id:
                G.add_node(venue_id, type='venue', label=venue_name)
                G.add_edge(paper_id, venue_id, relation='published_in')

            keywords = paper.get('keywords', [])
            for kw in keywords:
                G.add_node(kw, type='keyword', label=kw)
                G.add_edge(paper_id, kw, relation='has_keyword')

            references = paper.get('references', [])
            for ref_id in references:
                G.add_node(ref_id, type='paper', label=ref_id)
                G.add_edge(paper_id, ref_id, relation='cites')

            G.nodes[paper_id]['n_citation'] = paper.get('n_citation', 0)

    return G


def visualize_graph(G: nx.Graph, output_file: str = 'knowledge_graph.png') -> None:
    """
    可视化图。如果图太大，考虑简化（如移除关键词节点）。
    """
    if len(G.nodes) > 500:  # 阈值，防止太大
        print(f"Graph too large ({len(G.nodes)} nodes). Consider simplifying.")
        # 可选：移除关键词节点
        keywords = [n for n, d in G.nodes(data=True) if d.get('type') == 'keyword']
        G.remove_nodes_from(keywords)

    node_colors = []
    for node in G.nodes(data=True):
        node_type = node[1].get('type', 'unknown')
        if node_type == 'author':
            node_colors.append('skyblue')
        elif node_type == 'paper':
            node_colors.append('lightgreen')
        elif node_type == 'venue':
            node_colors.append('orange')
        elif node_type == 'keyword':
            node_colors.append('pink')
        else:
            node_colors.append('gray')

    pos = nx.spring_layout(G, seed=42, iterations=20)  # 减少迭代次数，加速布局
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_color=node_colors, node_size=500, font_size=8, edge_color='gray')

    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('Knowledge Graph for Author')
    plt.savefig(output_file)
    plt.show()
