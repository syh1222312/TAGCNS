# -*- coding: utf-8 -*-
"""
终极鲁棒 KG 划分（节点+边属性全保留）
按边 8:1:1 划分 → 输出 train/val/test.graphml
"""

import os
import re
import networkx as nx
from sklearn.model_selection import train_test_split
import logging
from collections import defaultdict

# ====================== 配置 ======================
INPUT_GRAPHML = "graph/knowledge_graph.graphML"
OUTPUT_DIR = "graph/splits_full"
RANDOM_STATE = 42
# ================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# ------------------- 1. 流式提取节点和边 -------------------
def extract_graph_data(file_path):
    """流式提取 <node> 和 <edge>（含所有 <data> 属性）"""
    log.info("正在流式提取节点和边（含所有属性）...")

    node_pattern = re.compile(r'<node[^>]*id=["\']([^"\']+)["\'][^>]*>(.*?)</node>', re.DOTALL)
    edge_pattern = re.compile(r'<edge[^>]*source=["\']([^"\']+)["\'][^>]*target=["\']([^"\']+)["\'][^>]*>(.*?)</edge>', re.DOTALL)
    data_pattern = re.compile(r'<data\s+key=["\']([^"\']+)["\']>([^<]*)</data>')

    nodes = {}  # node_id -> {key: value}
    edges = []  # (u, v, {key: value})

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        chunk = ""
        for line in f:
            chunk += line
            if len(chunk) > 10_000_000:  # 每10MB处理一次
                _process_chunk(chunk, node_pattern, edge_pattern, data_pattern, nodes, edges)
                chunk = "\n".join(chunk.split("\n")[-200:])  # 保留最后200行防跨行
        if chunk:
            _process_chunk(chunk, node_pattern, edge_pattern, data_pattern, nodes, edges)

    log.info(f"提取完成: {len(nodes):,} 个节点, {len(edges):,} 条边")
    return nodes, edges


def _process_chunk(chunk, node_pat, edge_pat, data_pat, nodes, edges):
    # 提取节点
    for node_id, inner in node_pat.findall(chunk):
        node_id = node_id.strip()
        if node_id not in nodes:
            nodes[node_id] = {}
        for key, val in data_pat.findall(inner):
            nodes[node_id][key] = val.strip()

    # 提取边
    for src, tgt, inner in edge_pat.findall(chunk):
        src, tgt = src.strip(), tgt.strip()
        edge_attrs = {}
        for key, val in data_pat.findall(inner):
            edge_attrs[key] = val.strip()
        edges.append((src, tgt, edge_attrs))


# ------------------- 2. 构建完整图 -------------------
def build_graph(nodes, edges):
    G = nx.MultiDiGraph()
    log.info("正在构建图结构...")

    # 添加节点
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    # 添加边
    for i, (u, v, attrs) in enumerate(edges):
        G.add_edge(u, v, **attrs)
        if i % 200_000 == 0 and i > 0:
            log.info(f"  已添加 {i:,} 条边...")

    log.info(f"图构建完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
    return G


# ------------------- 3. 按边划分 -------------------
def split_by_edges(G):
    edge_list = list(G.edges(data=True))  # (u, v, data_dict)
    total = len(edge_list)
    log.info(f"总边数: {total:,}")

    train_edges, temp = train_test_split(edge_list, train_size=0.8, random_state=RANDOM_STATE)
    val_edges, test_edges = train_test_split(temp, train_size=0.5, random_state=RANDOM_STATE)

    log.info(f"Train: {len(train_edges):,}, Val: {len(val_edges):,}, Test: {len(test_edges):,}")
    return train_edges, val_edges, test_edges


# ------------------- 4. 构建子图（保留所有节点属性） -------------------
def build_subgraph(G, edge_list):
    H = nx.MultiDiGraph()
    node_set = set()
    for u, v, _ in edge_list:
        node_set.update([u, v])

    # 添加节点（带完整属性）
    for n in node_set:
        H.add_node(n, **G.nodes[n])

    # 添加边（带完整属性）
    for u, v, data in edge_list:
        H.add_edge(u, v, **data)

    return H


# ------------------- 5. 保存子图 -------------------
def save_subgraph(G, name):
    path = f"{OUTPUT_DIR}/{name}.graphml"
    nx.write_graphml(G, path)
    log.info(f"{name}.graphml 已保存 → {path} (节点: {G.number_of_nodes():,}, 边: {G.number_of_edges():,})")


# ------------------- 主流程 -------------------
def main():
    if not os.path.exists(INPUT_GRAPHML):
        log.error(f"文件不存在: {INPUT_GRAPHML}")
        return

    # 1. 提取
    nodes, edges = extract_graph_data(INPUT_GRAPHML)

    # 2. 构建图
    G = build_graph(nodes, edges)

    # 3. 划分
    train_edges, val_edges, test_edges = split_by_edges(G)

    # 4. 构建子图
    log.info("正在构建子图...")
    train_G = build_subgraph(G, train_edges)
    val_G   = build_subgraph(G, val_edges)
    test_G  = build_subgraph(G, test_edges)

    # 5. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_subgraph(train_G, "train")
    save_subgraph(val_G,   "val")
    save_subgraph(test_G,  "test")

    log.info("所有子图划分完成！输出目录：graph/splits_full/")


if __name__ == "__main__":
    main()