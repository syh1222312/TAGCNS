import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import re


def load_graphml_robust(file_path):
    """鲁棒地加载GraphML，跳过错误部分"""
    print(f"鲁棒加载GraphML: {file_path}")

    edges = []
    nodes = set()

    try:
        # 方法1：尝试直接解析
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"XML解析错误: {e}")
        print("尝试分块解析...")

        # 方法2：分块读取并修复
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 清理非法XML字符
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e2:
            print(f"清理后仍解析失败: {e2}")
            # 方法3：只提取edge部分
            print("提取edge标签内容...")
            edge_pattern = r'<edge[^>]*source="([^"]*)"[^>]*target="([^"]*)"[^>]*>.*?<data key="d4">([^<]*)</data>.*?</edge>'
            matches = re.findall(edge_pattern, content, re.DOTALL | re.MULTILINE)

            for source, target, relation in matches:
                edges.append({
                    'source': source.strip(),
                    'target': target.strip(),
                    'relation': relation.strip()
                })

            print(f"通过正则提取到 {len(edges)} 条边")
            return edges, len(edges)

    # 正常解析流程
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

    # 提取所有edge
    for edge_elem in root.findall('.//graphml:edge', ns):
        source = edge_elem.get('source', '')
        target = edge_elem.get('target', '')

        # 查找关系类型
        relation = 'unknown'
        for data_elem in edge_elem.findall('graphml:data', ns):
            if data_elem.get('key') == 'd4':
                relation = data_elem.text or 'unknown'
                break

        if source and target:
            edges.append({
                'source': source.strip(),
                'target': target.strip(),
                'relation': relation.strip()
            })
            nodes.add(source)
            nodes.add(target)

    print(f"成功解析 {len(edges)} 条边，{len(nodes)} 个节点")
    return edges, len(nodes)


def extract_source_target_pairs(edges):
    """创建source-target对"""
    source_target_pairs = defaultdict(list)

    for edge in edges:
        s = edge['source']
        q = edge['target']
        relation = edge['relation']

        source_target_pairs[(s, q)].append({
            'relation': relation,
            'edge_id': f"{s}-{q}-{relation}"
        })

    pairs = [{'s': s, 'q': q, 'edges': edge_list}
             for (s, q), edge_list in source_target_pairs.items()]

    print(f"唯一(s, q)对数: {len(pairs)}")
    return pairs


def split_pairs(pairs, test_size=0.1, val_size=0.1):
    """修正的8:1:1划分"""
    total_pairs = len(pairs)
    print(f"总(s,q)对数: {total_pairs}")

    # 先划分训练+验证 vs 测试 (90:10)
    train_val_pairs, test_pairs = train_test_split(
        pairs, test_size=test_size, random_state=42
    )

    # 再从训练+验证中划分训练 vs 验证 (8/9 : 1/9 ≈ 80:10 of total)
    train_pairs, val_pairs = train_test_split(
        train_val_pairs,
        test_size=val_size / (1 - test_size),  # 0.1 / 0.9 ≈ 0.111
        random_state=42
    )

    # 提取边
    def get_edges(pairs_list):
        return {edge['edge_id'] for pair in pairs_list
                for edge in pair['edges']}

    train_edges = get_edges(train_pairs)
    val_edges = get_edges(val_pairs)
    test_edges = get_edges(test_pairs)

    # 验证比例
    train_count = len(train_pairs)
    val_count = len(val_pairs)
    test_count = len(test_pairs)

    print(f"训练集: {train_count} 对 ({train_count / total_pairs * 100:.1f}%)")
    print(f"验证集: {val_count} 对 ({val_count / total_pairs * 100:.1f}%)")
    print(f"测试集: {test_count} 对 ({test_count / total_pairs * 100:.1f}%)")

    # 严格检查
    assert abs(train_count - total_pairs * 0.8) <= total_pairs * 0.01, "训练集比例错误"
    assert abs(val_count - total_pairs * 0.1) <= total_pairs * 0.01, "验证集比例错误"
    assert abs(test_count - total_pairs * 0.1) <= total_pairs * 0.01, "测试集比例错误"

    return {
        'train_pairs': train_pairs,
        'val_pairs': val_pairs,
        'test_pairs': test_pairs,
        'train_edges': train_edges,
        'val_edges': val_edges,
        'test_edges': test_edges
    }


def save_results(splits, output_dir):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    splits_info = ['train', 'val', 'test']

    for split_type in splits_info:
        # 保存(s,q)对
        pairs_file = os.path.join(output_dir, f'{split_type}_pairs.txt')
        with open(pairs_file, 'w') as f:
            for pair in splits[f'{split_type}_pairs']:
                f.write(f"{pair['s']}\t{pair['q']}\n")

        # 保存边
        edges_file = os.path.join(output_dir, f'{split_type}_edges.txt')
        with open(edges_file, 'w') as f:
            for edge_id in sorted(splits[f'{split_type}_edges']):
                f.write(f"{edge_id}\n")

    # 统计
    total_pairs = sum(len(splits[k]) for k in ['train_pairs', 'val_pairs', 'test_pairs'])
    print(f"✅ 划分完成！总计 {total_pairs} 对，比例 8:1:1")
    print(f"结果保存在: {output_dir}")


def main():
    graphml_file = "../graph/knowledge_graph.graphml"
    output_dir = "graph/splits"

    if not os.path.exists(graphml_file):
        print(f"❌ 文件不存在: {graphml_file}")
        return

    # 鲁棒加载
    edges, node_count = load_graphml_robust(graphml_file)

    if not edges:
        print("❌ 未提取到边")
        return

    # 创建(s,q)对
    pairs = extract_source_target_pairs(edges)

    # 划分
    splits = split_pairs(pairs)

    # 保存
    save_results(splits, output_dir)


if __name__ == "__main__":
    main()