import networkx as nx
import torch
import os


# ==================== 1. 读取 GraphML ====================
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

# ==================== 2. 构建索引 ====================
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

# ==================== 3. 初始化嵌入 & 参数 ====================
def init_embeddings_and_params(entity_to_idx, relation_to_idx, dim=128, hidden=64):
    h_entities = torch.randn(len(entity_to_idx), dim)
    h_relations = torch.randn(len(relation_to_idx), dim)

    # 一阶注意力
    W1 = torch.randn(3 * dim, hidden)
    U1 = torch.randn(hidden, 1)

    # 二阶注意力：agg_one.T @ W2 @ [h_r, h_o2]
    W2 = torch.randn(dim, 2 * dim)  # (d, 2d)

    return h_entities, h_relations, W1, U1, W2


# ==================== 4. 核心聚合邻居====================
def aggregate_one_and_two_hop_vectors(
        head_id, G, Max,
        h_entities, h_relations,
        entity_to_idx, relation_to_idx,
        W1, U1, W2,
        activate=torch.nn.LeakyReLU(0.2)  # 仅用于一阶
):
    """
    返回：
        agg_one: 一阶邻居聚合向量（含激活）
        agg_two: 二阶邻居聚合向量（无激活，线性打分）
    """
    if head_id not in entity_to_idx:
        print(f"[Warning] Node {head_id} not in graph.")
        dim = h_entities.size(1)
        zero = torch.zeros(dim)
        return zero, zero

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
        zero = torch.zeros(dim)
        return zero, zero

    # --- Step 2: 一阶注意力（含激活）---
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

    # --- Step 3: 二阶注意力（无激活）---
    if not two_hop_candidates:
        return agg_one, torch.zeros(dim)

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

    return agg_one, agg_two;
