import torch
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def auto_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def jsonify_tensors(batch):
    for result in batch:
        for node in result['nodes']:
            for key, value in node.items():
                if isinstance(value, torch.Tensor):
                    node[key] = value.item()
    return batch
