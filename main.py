import anchor
if __name__ == "__main__":
   file_path = 'data/DBLP/dblp-v12-author.json'
   target_author_id = '2115010573'  # Replace with your target author ID
   graph = anchor.generate_knowledge_graph(file_path, target_author_id)
   anchor.visualize_graph(graph)