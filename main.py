import anchor
sorted_graphs = anchor.generate_temporal_knowledge_graphs('data/DBLP/dblp-v12-author.json', '67597021')
anchor.visualize_temporal_graphs(sorted_graphs)