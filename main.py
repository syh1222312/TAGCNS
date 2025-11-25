# 假设 generate_knowledge_graph 已定义在当前文件中，或从 anchor.py 导入
# from anchor import generate_knowledge_graph  # 如果在其他模块
import anchor
if __name__ == "__main__":
    file_path = 'data/DBLP/dblp-v12-author.json'  # 你的JSON文件路径
    target_author_id = '67597021'  # 替换为你的目标作者ID，例如从示例中的 '53f42f36dabfaedce54dcd0c'
    graph = anchor.generate_knowledge_graph(file_path, target_author_id)

    # 可选：可视化图谱（需要额外代码，如之前提供的 visualize_graph）
    anchor.visualize_graph(graph)