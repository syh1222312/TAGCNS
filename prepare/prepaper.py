from decimal import Decimal
import json
import ijson


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)


def is_empty(value):
    if isinstance(value, str):
        return value == ''
    elif isinstance(value, list):
        return len(value) == 0
    elif isinstance(value, dict):
        return len(value) == 0
    return False


# File paths
input_file = '../data/DBLP/dblp-v12.json'
output_file = '../data/DBLP/dblp-v12-paper.json'

seen_ids = set()
unique_authors = set()
num_papers = 0

with open(output_file, 'w', encoding='utf-8') as out:
    out.write('[')
    first = True
    with open(input_file, 'rb') as f:
        for paper in ijson.items(f, 'item'):
            # Clean the paper: remove empty fields
            cleaned_paper = {}
            for k, v in paper.items():
                if not is_empty(v):
                    cleaned_paper[k] = v

            # Check if 'id' present and not empty (but since str, if '' skipped above)
            if 'id' in cleaned_paper and cleaned_paper['id'] not in seen_ids:
                seen_ids.add(cleaned_paper['id'])
                # Check authors >5
                if 'authors' in cleaned_paper and isinstance(cleaned_paper['authors'], list) and len(
                        cleaned_paper['authors']) > 5:
                    if not first:
                        out.write(',')
                    json.dump(cleaned_paper, out, ensure_ascii=False, cls=DecimalEncoder)
                    first = False
                    num_papers += 1
                    # Collect unique authors
                    for author in cleaned_paper['authors']:
                        if 'id' in author:
                            unique_authors.add(author['id'])

    out.write(']')

print(f"Number of papers: {num_papers}")
print(f"Number of unique authors: {len(unique_authors)}")

