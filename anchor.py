import json
import networkx as nx
import matplotlib.pyplot as plt

# 加载数据
file_path = "/Users/sheyunhan/PycharmProjects/TAGCN/data/DBLP/dblp-v12-author.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 指定目标实体
entity_name = "Tomohisa Yamashita"
entity_id = 2139313784  # 从提供的示例中获取

# 创建有向图
G = nx.DiGraph()

# 第一步：查找作者并添加节点
related_papers = []
for paper in data:
    authors = paper.get('authors', [])
    for author in authors:
        author_id = author.get('id')
        author_name = author.get('name')
        if author_id == entity_id or author_name == entity_name:
            related_papers.append(paper)
            break

if not related_papers:
    raise ValueError("未找到指定的作者")

# 添加中心作者节点
G.add_node(entity_name, type="Author", id=entity_id)

# 第二步：为每个相关论文添加节点和边
for paper in related_papers:
    paper_id = paper.get('id')
    paper_title = paper.get('title')
    G.add_node(paper_title, type="Paper", id=paper_id, year=paper.get('year'))
    G.add_edge(entity_name, paper_title, relation="authored")

    # 添加合作作者
    authors = paper.get('authors', [])
    for author in authors:
        co_author_id = author.get('id')
        if co_author_id != entity_id:
            co_author_name = author.get('name')
            if not G.has_node(co_author_name):
                G.add_node(co_author_name, type="Author", id=co_author_id)
            G.add_edge(entity_name, co_author_name, relation="co-authored with")
            G.add_edge(co_author_name, paper_title, relation="authored")

    # 添加venue
    venue = paper.get('venue', {})
    venue_raw = venue.get('raw')
    if venue_raw:
        venue_id = venue.get('id')
        G.add_node(venue_raw, type="Venue", id=venue_id)
        G.add_edge(paper_title, venue_raw, relation="published in")

    # 添加fields of study (fos)
    fos_list = paper.get('fos', [])
    for fos in fos_list:
        fos_name = fos.get('name')
        if fos_name:
            G.add_node(fos_name, type="Field of Study")
            G.add_edge(paper_title, fos_name, relation="related to")

    # 如果有references，添加（假设references是list of ids）
    references = paper.get('references', [])
    for ref_id in references:
        # 查找引用论文是否在数据中
        ref_paper = next((p for p in data if p.get('id') == ref_id), None)
        if ref_paper:
            ref_title = ref_paper.get('title')
            G.add_node(ref_title, type="Paper", id=ref_id)
            G.add_edge(paper_title, ref_title, relation="cites")

# 绘制知识图谱
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8, font_weight='bold')

# 添加边标签
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title(f"Knowledge Graph for Author: {entity_name}")
plt.savefig("knowledge_graph.png")  # 保存为文件，而不是显示