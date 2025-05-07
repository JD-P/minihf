import os
import time
import asyncio
from typing import List, Dict, Any
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from threading import Thread
from queue import Queue, Empty
import uvicorn

app = FastAPI()

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = "modern-bert"
    encoding_format: str = "float"
    user: str | None = None

class EmbeddingProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = AutoModelForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base",
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        ).to(self.device).eval()
        
        self.max_batch_size = 4
        self.batch_timeout = 0.1
        self.queue = Queue()
        self.stop_event = False
        self.process_thread = Thread(target=self.process_batches, daemon=True)
        self.process_thread.start()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            if len(last_hidden.shape) == 2:
                last_hidden = last_hidden.unsqueeze(0)
            attention_mask = inputs.attention_mask.unsqueeze(-1)

            embeddings = last_hidden[:, -1, :]
            # Mean pooling
            #numerator = (last_hidden * attention_mask).sum(dim=1)
            #denominator = attention_mask.sum(dim=1)
            #denominator = torch.clamp(denominator, min=1e-9)
            #embeddings = numerator / denominator

            # L2 Normalization
            #embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Clean numerical values
            embeddings = embeddings.cpu().to(torch.float32)
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            
            return embeddings.tolist()

    def process_batches(self):
        while not self.stop_event:
            batch = []
            start_time = time.time()
            
            while len(batch) < self.max_batch_size and (time.time() - start_time) < self.batch_timeout:
                try:
                    item = self.queue.get_nowait()
                    batch.append(item)
                except Empty:
                    time.sleep(0.001)
            
            if batch:
                try:
                    texts = [item["text"] for item in batch]
                    embeddings = self.generate_embeddings(texts)
                    
                    for item, emb in zip(batch, embeddings):
                        item["loop"].call_soon_threadsafe(
                            item["future"].set_result,
                            {
                                "embedding": emb,
                                "token_count": len(item["tokens"])
                            }
                        )
                except Exception as e:
                    for item in batch:
                        item["loop"].call_soon_threadsafe(
                            item["future"].set_exception,
                            e
                        )

    def shutdown(self):
        self.stop_event = True
        self.process_thread.join()

processor = EmbeddingProcessor()

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    if request.model != "modern-bert":
        raise HTTPException(400, detail="Only 'modern-bert' model is supported")
    
    inputs = [request.input] if isinstance(request.input, str) else request.input
    futures = []

    for text in inputs:
        if not text.strip():
            raise HTTPException(400, detail="Input text cannot be empty")
        
        future = asyncio.Future()
        tokens = processor.tokenizer.encode(text, truncation=False)
        
        if len(tokens) > 8192:
            tokens = tokens[-8192:]
            text = processor.tokenizer.decode(tokens)
        
        processor.queue.put({
            "future": future,
            "text": text,
            "tokens": tokens,
            "loop": asyncio.get_running_loop()
        })
        futures.append(future)
    
    results = await asyncio.gather(*futures, return_exceptions=True)
    
    embeddings = []
    total_tokens = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise HTTPException(500, detail=f"Embedding generation failed: {str(result)}")
        
        embeddings.append({
            "object": "embedding",
            "embedding": result["embedding"],
            "index": i
        })
        total_tokens += result["token_count"]
    
    return {
        "object": "list",
        "data": embeddings,
        "model": "modern-bert",
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

@app.on_event("shutdown")
async def shutdown_event():
    processor.shutdown()

if __name__ == "__main__":
    uvicorn.run(app,
                host="0.0.0.0",
                port=int(os.getenv("PORT", 5002)),
                log_level="debug")
