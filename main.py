from fspm import extract_st_pairs_from_file,remove_invalid_chars

# 示例调用：替换为实际的 graphml 文件路径
graphml_path = 'graph/splits_full/train.graphml'  # 替换为实际路径
tmp=remove_invalid_chars(graphml_path)
result = extract_st_pairs_from_file(graphml_path)

# 打印或处理结果（假设函数返回某些值）
print(len(result))