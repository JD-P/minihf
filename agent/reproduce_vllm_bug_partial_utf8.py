import random
import json
import requests

prompts = requests.get("https://minihf.com/vllm_utf8_logprobs_error_reproduce_prompts.json").json()

port = 5001
n = 1
model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

payload = {
    "n": n,
    "temperature": 1,
    "top_k": 50,
    "repetition_penalty": 1.02,
    "max_tokens": 1,
    "model": model_name,
    "prompt": prompts,
    "stream": False,
    "logprobs": 100,
    "seed": random.randrange(1000000)
}

print("With logprobs = 100")
print(requests.post(f"http://localhost:{port}/v1/completions/", json=payload).json(), end="\n\n")

payload["logprobs"] = 0
print("With logprobs = 0")
print(requests.post(f"http://localhost:{port}/v1/completions/", json=payload).json(), end="\n\n")

no_unicode = prompts[0].replace("\u2019", "'").replace("\U0001f642", ":)").replace("\u201c", '').replace("\u201d", '"').replace("\u2014", "-")
assert no_unicode.encode("ascii")
prompts2 = [no_unicode,]
payload["logprobs"] = 100
payload["prompts"] = prompts2
print("With no unicode in input prompt string")
print(requests.post(f"http://localhost:{port}/v1/completions/", json=payload).json(), end="\n\n")
