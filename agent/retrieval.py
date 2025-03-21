import os
import time
from sqlite3 import Connection as SQLite3Connection
import aiosqlite
import sqlite_vec
import aiohttp
import json
import torch
from queue import Queue
from transformers import AutoTokenizer, AutoModelForMaskedLM
from weave import generate_outputs_vllm, evaluate_outputs_vllm

class ModernBertRag:
    def __init__(self, weave_tree, db_path="blocks.db", port=5001, embed_port=5002):
        self.db_path = db_path
        self.tree = weave_tree
        self.model_name = weave_tree.model_name
        self.port = port
        self.embed_port = embed_port
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.queue = Queue()
        
    async def setup(self):
        conn = await self._connect()
                
        # Create tables
        cursor = await conn.cursor()
        await cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS block_embeddings USING vec0(rowid INTEGER PRIMARY KEY, embedding FLOAT[768])
        """)
        await cursor.execute("""
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
        await cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocks_rowid ON blocks(rowid);")
        await conn.commit()
        await cursor.close()
        await conn.close()

    async def _connect(self):
        """Async connection with sqlite_vec extension loading"""
        conn = await aiosqlite.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False
        )
        
        # Access raw SQLite3 connection to load extensions
        raw_conn: SQLite3Connection = conn._connection
        
        # Run blocking extension loading in executor
        raw_conn.enable_load_extension(True)
        sqlite_vec.load(raw_conn)
        raw_conn.enable_load_extension(False)
        
        # Critical performance settings
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA busy_timeout=5000;")
        
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
        inputs = self.tokenizer(combined_text,
                                return_tensors="pt",
                                add_special_tokens=False,
                                truncation=False)
        tokens = inputs["input_ids"][0][-8192:]  # Take last 8192 tokens

        # Decode tokens back to text
        decoded_text = self.tokenizer.decode(tokens)
        
        # Generate embedding
        async with aiohttp.ClientSession() as session:
            payload = {"input":decoded_text,
                       "model": "modern-bert"}
            async with session.post(f"http://localhost:{self.embed_port}/v1/embeddings",
                                    json=payload) as response:
                response_json = await response.json()
                assert len(response_json["data"]) == 1
                embedding = response_json["data"][0]["embedding"]

        # Insert into SQLite
        conn = await self._connect()
        cursor = await conn.cursor()
        await cursor.execute("""
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
        await cursor.execute("""
            INSERT INTO block_embeddings (rowid, embedding)
            VALUES (?, json(?))
        """, (rowid, json.dumps(embedding)))
        
        await conn.commit()
        await cursor.close()
        await conn.close()

        # Update summary tree
        if hasattr(self, 'tree'):
            self.tree.add_summary((block_id, full_description))

        print(f"Embedded block {block_id}")
        return block_id

    async def search(self, text, limit=5, before=time.time()):
        """Search for similar blocks"""
        # Generate query embedding
        print("Tokenizing input for search...")
        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                truncation=False,
                                add_special_tokens=False)
        truncated_text = self.tokenizer.decode(inputs["input_ids"][0][-8192:])
        inputs = self.tokenizer(truncated_text, return_tensors="pt", truncation=False)
        print("Embedding input for search...")
        async with aiohttp.ClientSession() as session:
            payload = {"input":truncated_text,
                       "model": "modern-bert"}
            async with session.post(f"http://localhost:{self.embed_port}/v1/embeddings",
                                    json=payload) as response:
                response_json = await response.json()
                assert len(response_json["data"]) == 1
                query_embedding = response_json["data"][0]["embedding"]

        # Execute KNN search
        print("Connecting to sqlite...")
        conn = await self._connect()
        cursor = await conn.cursor()
        print("Executing search...")
        await cursor.execute("""
            SELECT blocks.*, block_embeddings.distance
            FROM block_embeddings
            JOIN blocks ON block_embeddings.rowid = blocks.rowid
            WHERE block_embeddings.embedding MATCH json(?)
            AND blocks.timestamp < ?
            AND k = ?
            ORDER BY distance
            
        """, (json.dumps(query_embedding), before, limit))
        print("Fetching results...")
        results = []
        for row in await cursor.fetchall():
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
