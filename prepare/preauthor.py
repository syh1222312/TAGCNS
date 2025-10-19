from decimal import Decimal
import json
from collections import defaultdict
import ijson

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)

# File paths
input_file = '../data/DBLP/dblp-v12-paper.json'
output_file = '../data/DBLP/dblp-v12-author.json'

# First pass: Count authors with >=10 papers in this dataset
author_count = defaultdict(int)
with open(input_file, 'rb') as f:
    for paper in ijson.items(f, 'item'):
        if 'authors' in paper:
            for author in paper['authors']:
                if 'id' in author:
                    author_count[author['id']] += 1

qualified_authors = {auth for auth, count in author_count.items() if count >= 10}

# Second pass: Write file with papers that have at least one qualified author
with open(output_file, 'w', encoding='utf-8') as out:
    out.write('[')
    first = True
    with open(input_file, 'rb') as f:
        for paper in ijson.items(f, 'item'):
            if 'authors' in paper and any('id' in a and a['id'] in qualified_authors for a in paper['authors']):
                if not first:
                    out.write(',')
                json.dump(paper, out, ensure_ascii=False, cls=DecimalEncoder)
                first = False
    out.write(']')