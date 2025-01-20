import os
import json
import hashlib
import tantivy
from tantivy import Index, SchemaBuilder

schema_builder = SchemaBuilder()
schema_builder.add_text_field("id", stored=True, tokenizer_name='raw')
schema_builder.add_text_field("type", stored=True)
schema_builder.add_text_field("render", stored=True)
schema_builder.add_text_field("q", stored=True)
schema_builder.add_float_field("score", stored=True)
schema_builder.add_integer_field("index", stored=True)
schema_builder.add_float_field("timestamp", stored=True)
schema_builder.add_text_field("description", stored=True)

bm25_schema = schema_builder.build()

if not os.path.exists("memories"):
    os.mkdir("memories")
if not os.path.exists("memories/bm25"):
    os.mkdir("memories/bm25")
bm25_index = Index(bm25_schema, path="./memories/bm25")

example_filenames = [filename for filename in
                     os.listdir("./bootstraps/example_blocks")
                     if filename.endswith(".json")]
example_blocks = []
for filename in example_filenames:
    with open("./bootstraps/example_blocks/" + filename) as infile: 
        block_metadata = json.load(infile)
    with open("./bootstraps/example_blocks/" + filename[:-5] + ".py") as infile:
        block_render = infile.read()
    block_metadata["render"] = block_render
    example_blocks.append(block_metadata)

for example in example_blocks:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(example["render"].encode('utf-8'))
    hash_hex = sha256_hash.hexdigest()
    
    bm25_query = f"id:'{hash_hex}'"
    searcher = bm25_index.searcher()
    query = bm25_index.parse_query(bm25_query, ["id",])
    results = searcher.search(query, limit=25).hits
    retrieved_blocks = [searcher.doc(result[1]) for result in results]
    if retrieved_blocks:
        print(f"Skipped block {hash_hex}")
        continue
    
    writer = bm25_index.writer()
    writer.add_document(tantivy.Document(
        id=hash_hex,
        type=example["type"],
        render=example["render"],
        q=example["q"],
        score=example["score"],
        index=example["index"],
        timestamp=example["timestamp"],
        description=example["description"],
    ))
    writer.commit()
    print(f"Wrote block {hash_hex}")




