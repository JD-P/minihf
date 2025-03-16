import os
import sqlite3
import sqlite_vec
import json
import torch
from queue import Queue
from transformers import AutoTokenizer, AutoModelForMaskedLM
from weave import generate_outputs_vllm, evaluate_outputs_vllm

class ModernBertRag:
    def __init__(self, weave_tree, db_path="blocks.db", port=5001):
        self.db_path = db_path
        self.tree = weave_tree
        self.model_name = weave_tree.model_name
        self.port = port
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
        self.queue = Queue()
        
        conn = self._connect()
                
        # Create tables
        cursor = conn.cursor()
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS block_embeddings USING vec0(rowid INTEGER PRIMARY KEY, embedding FLOAT[768])
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                rowid INTEGER PRIMARY KEY,
                block_id TEXT,
                type TEXT,
                render TEXT,
                q TEXT,
                score FLOAT,
                _index INTEGER,
                timestamp FLOAT,
                description TEXT,
                embed_context TEXT
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()

    def _connect(self):
        # Initialize SQLite connection with sqlite-vec
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn
    
    def add(self, item):
        """Add item to processing queue"""
        assert item["id"]
        assert item["render"]
        assert "context" in item
        self.queue.put(item)

    async def process_item(self):
        """Process one item from the queue"""
        if self.queue.empty():
            return None

        item = self.queue.get(block=False)
        render = item["render"]
        context = item["context"]
        block_id = item["id"]

        """
        if "description" not in item:
            if not os.path.exists("/app/templates/describe1.txt"):
                describe1_path = "templates/describe1.txt"
            else:
                describe1_path = "/app/templates/describe1.txt"
            if not os.path.exists("/app/templates/describe2.txt"):
                describe2_path = "templates/describe2.txt"
            else:
                describe2_path = "/app/templates/describe2.txt"
            # Generate descriptions
            with open(describe1_path) as infile:
                template = infile.read()
                prompt = template.format(rendered_block=render)
                object_description = generate_outputs_vllm(
                    self.model_name, prompt, 512, port=self.port, n=1, stop=["</summary>"]
                )[0]

            with open(describe2_path) as infile:
                template = infile.read()
                prompt = template.format(
                    rendered_block=render,
                    object_description=object_description,
                    rendered_context=context
                )
                context_description = generate_outputs_vllm(
                    self.model_name, prompt, 512, port=self.port, n=1, stop=["</summary>"]
                )[0]

            full_description = f"{object_description}\n\n{context_description}"
        else:
            full_description = item["description"]
        """
        full_description = ""
        
        # Tokenize and process text
        combined_text = f"{context}\n{render}\n\n{full_description}"
        inputs = self.tokenizer(combined_text, return_tensors="pt", truncation=False)
        tokens = inputs["input_ids"][0][-8192:]  # Take last 8192 tokens
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(tokens.unsqueeze(0), output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embedding = last_hidden.mean(dim=1).squeeze().tolist()

        # Decode tokens back to text
        decoded_text = self.tokenizer.decode(tokens)

        # Insert into SQLite
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO blocks (
                block_id, type, render, q, score, _index, timestamp, description, embed_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            block_id,
            item.get("type"),
            render,
            item.get("q"),
            item.get("score"),
            item.get("_index"),
            item.get("timestamp"),
            full_description,
            decoded_text
        ))
        
        # Insert embedding using same rowid
        rowid = cursor.lastrowid
        cursor.execute("""
            INSERT INTO block_embeddings (rowid, embedding)
            VALUES (?, json(?))
        """, (rowid, json.dumps(embedding)))
        
        conn.commit()
        cursor.close()
        conn.close()

        # Update summary tree
        if hasattr(self, 'tree'):
            self.tree.add_summary((block_id, full_description))

        print(f"Embedded block {block_id}")
        return block_id

    def search(self, text, limit=5):
        """Search for similar blocks"""
        # Generate query embedding
        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                truncation=False,
                                add_special_tokens=False)
        truncated_text = self.tokenizer.decode(inputs["input_ids"][0][-8192:])
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            query_embedding = last_hidden.mean(dim=1).squeeze().tolist()

        # Execute KNN search
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT blocks.*, block_embeddings.distance
            FROM block_embeddings
            JOIN blocks ON block_embeddings.rowid = blocks.rowid
            WHERE block_embeddings.embedding MATCH json(?)
            AND k = ?
            ORDER BY distance
            
        """, (json.dumps(query_embedding), limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[1],
                "type": row[2],
                "render": row[3],
                "q": row[4],
                "score": row[5],
                "index": row[6],
                "timestamp": row[7],
                "description": row[8],
                "embed_context": row[9],
                "distance": row[10]
            })
        
        return results
