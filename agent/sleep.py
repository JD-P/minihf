import os
import json
import hashlib
import asyncio
from retrieval import ModernBertRag  # Assuming your ModernBertRag is in this module
from argparse import ArgumentParser

class MockWeaveAgentTree:
    """Mock tree that just tracks summaries in memory"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.summaries = []
    
    def add_summary(self, summary):
        self.summaries.append(summary)
        print(f"Added summary: {summary[0]}")

async def bootstrap_rag_memories(model_name):
    # Initialize mock tree and RAG system
    mock_tree = MockWeaveAgentTree(model_name)
    rag = ModernBertRag(mock_tree, db_path="blocks.db")
    await rag.setup()
    
    # Load example blocks
    example_dir = "./bootstraps/example_blocks"
    example_files = [f for f in os.listdir(example_dir) if f.endswith(".json")]
    
    for filename in example_files:
        json_path = os.path.join(example_dir, filename)
        code_path = os.path.join(example_dir, filename[:-5] + ".py")
        
        with open(json_path) as f:
            metadata = json.load(f)
        with open(code_path) as f:
            render_content = f.read()
        
        # Create unique ID from render content
        sha = hashlib.sha256()
        sha.update(render_content.encode('utf-8'))
        block_id = sha.hexdigest()
        
        # Check if block already exists
        conn = await rag._connect()
        cursor = await conn.cursor()
        await cursor.execute("SELECT 1 FROM blocks WHERE block_id=?", (block_id,))
        exists = await cursor.fetchone() is not None
        await cursor.close()
        await conn.close()
        
        if exists:
            print(f"Block {block_id[:8]}... already exists, skipping")
            continue
        
        # Create the item structure ModernBERT-RAG expects
        rag_item = {
            "id": block_id,
            "render": render_content,
            "context": metadata.get("context", ""),
            "type": metadata.get("type", "code_block"),
            "q": metadata.get("q", ""),
            "score": metadata.get("score", 0.0),
            "_index": metadata.get("index", 0),
            "timestamp": metadata.get("timestamp", 0.0)
        }
        
        # Add to processing queue and process immediately
        rag.add(rag_item)
        processed_id = await rag.process_item()
        
        if processed_id:
            print(f"Successfully added memory block {processed_id[:8]}...")
        else:
            print(f"Failed to process block {block_id[:8]}...")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name")
    args = parser.parse_args()
    asyncio.run(bootstrap_rag_memories(args.model_name))
    print("Bootstrap memories added!")
