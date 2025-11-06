import os
import networkx as nx
import torch
import logging


def load_kg_graphml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"KG file not found: {file_path}")

    G = nx.read_graphml(file_path)

    # 检查空图
    if G.number_of_nodes() == 0:
        raise ValueError("Error: The graph is empty (no nodes).")
    if G.number_of_edges() == 0:
        raise ValueError("Error: The graph has no edges.")

    # 检查边属性
    for u, v, data in G.edges(data=True):
        if 'relation' not in data and 'd10' not in data:
            raise ValueError(f"Error: Edge ({u}, {v}) is missing 'relation' or 'd10' attribute.")
        rel = data.get('relation') or data.get('d10') or 'unknown'
        data['relation'] = rel

    print(f"[KG] Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_mappings(G):
    entities = list(G.nodes())
    if not entities:
        raise ValueError("Error: No entities (nodes) found in the graph.")

    entity_to_idx = {ent: i for i, ent in enumerate(entities)}
    relations = {data['relation'] for _, _, data in G.edges(data=True)}

    if not relations:
        raise ValueError("Error: No relations found in the graph edges.")

    relation_to_idx = {rel: i for i, rel in enumerate(sorted(relations))}

    print(f"[Mapping] Entities: {len(entities)}, Relations: {len(relations)}")
    return entity_to_idx, relation_to_idx


def init_embeddings_and_params(entity_to_idx, relation_to_idx, dim=128, hidden=64):
    h_entities = torch.randn(len(entity_to_idx), dim)
    h_relations = torch.randn(len(relation_to_idx), dim)

    # 一阶注意力
    W1 = torch.randn(3 * dim, hidden)
    U1 = torch.randn(hidden, 1)

    # 二阶注意力
    W2 = torch.randn(dim, 2 * dim)  # (d, 2d)

    # 门控机制参数：g = sigmoid(W3 @ agg_two + b)
    W3 = torch.randn(dim, 1)  # (dim, 1) 用于生成标量 g
    b = torch.randn(1)  # (1,) 标量偏置

    return h_entities, h_relations, W1, U1, W2, W3, b


def aggregate_one_and_two_hop_vectors(
        head_id, G, Max,
        h_entities, h_relations,
        entity_to_idx, relation_to_idx,
        W1, U1, W2, W3, b,
        activate=torch.nn.LeakyReLU(0.2),  # 仅用于一阶
        sigma=torch.sigmoid  # 用于门控和最终激活
):
    """
    修改说明：
    1. 实现公式：g = sigmoid(W3 @ agg_two + b)，f(e) = σ(g * agg_two + (1 - g) * agg_one)。
    2. 返回最终嵌入 f(e)，而不是 agg_two。
    3. 使用 sigma（默认为 sigmoid）作为门控和最终激活函数。
    返回：
        f_e: 最终嵌入向量 f(e)，形状为 (dim,)
    """
    if head_id not in entity_to_idx:
        print(f"[Warning] Node {head_id} not in graph.")
        dim = h_entities.size(1)
        return torch.zeros(dim)

    head_idx = entity_to_idx[head_id]
    h_head = h_entities[head_idx]
    dim = h_head.size(0)

    # --- Step 1: 收集一阶邻居 ---
    one_hop_triples = []  # (r_idx, o_idx)
    two_hop_candidates = []  # (r2_idx, o2_idx, o_id)

    for o in G.successors(head_id):
        r = G.edges[head_id, o]['relation']
        if r not in relation_to_idx or o not in entity_to_idx:
            continue
        r_idx = relation_to_idx[r]
        o_idx = entity_to_idx[o]
        one_hop_triples.append((r_idx, o_idx))

        o_id = next(k for k, v in entity_to_idx.items() if v == o_idx)
        if G.out_degree(o_id) < Max:
            for o2 in G.successors(o_id):
                r2 = G.edges[o_id, o2]['relation']
                if r2 not in relation_to_idx or o2 not in entity_to_idx:
                    continue
                r2_idx = relation_to_idx[r2]
                o2_idx = entity_to_idx[o2]
                two_hop_candidates.append((r2_idx, o2_idx, o_id))

    if not one_hop_triples:
        return torch.zeros(dim)

    # --- Step 2: 一阶注意力（含激活，用于二阶查询和最终嵌入） ---
    scores_one = []
    for r_idx, o_idx in one_hop_triples:
        h_r = h_relations[r_idx]
        h_o = h_entities[o_idx]
        cat = torch.cat([h_head, h_r, h_o])  # (3d,)
        logit = activate(cat @ W1 @ U1)  # 激活
        scores_one.append(logit)

    scores_one = torch.cat(scores_one)
    alphas_one = torch.softmax(scores_one, dim=0)

    agg_one = torch.zeros(dim)
    for alpha, (_, o_idx) in zip(alphas_one, one_hop_triples):
        agg_one += alpha.item() * h_entities[o_idx]

    # --- Step 3: 二阶注意力（无激活） ---
    if not two_hop_candidates:
        return sigma(agg_one)  # 若无二阶邻居，返回激活后的一阶嵌入

    scores_two = []
    two_hop_list = []

    for r2_idx, o2_idx, _ in two_hop_candidates:
        h_r2 = h_relations[r2_idx]
        h_o2 = h_entities[o2_idx]
        pair = torch.cat([h_r2, h_o2])  # (2d,)
        query = agg_one @ W2  # (1, 2d)
        logit = query @ pair  # (1,1) → 标量，无激活
        scores_two.append(logit)
        two_hop_list.append((r2_idx, o2_idx))

    scores_two = torch.cat(scores_two)  # (num_two,)
    alphas_two = torch.softmax(scores_two, dim=0)

    # --- Step 4: 聚合二阶邻居 ---
    agg_two = torch.zeros(dim)
    for alpha, (_, o2_idx) in zip(alphas_two, two_hop_list):
        agg_two += alpha.item() * h_entities[o2_idx]

    # --- Step 5: 门控机制和最终嵌入 ---
    g = sigma(agg_two @ W3 + b)  # g: 标量，sigmoid(W3 @ agg_two + b)
    f_e = sigma(g * agg_two + (1 - g) * agg_one)  # f(e) = σ(g * f2(s) + (1 - g) * f1(s))

    return f_e

