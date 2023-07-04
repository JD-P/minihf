import random
import json
import zipfile

class ZippedConversationsDataset:
    def __init__(self, zip_file):
        self.training_items = []
        zip_ = zipfile.ZipFile(zip_file)
        for file_ in zip_.namelist():
            if file_.endswith("/"): # Skip directories
                continue
            if file_.startswith("__MACOSX"): # Mac OS X adds garbage to zips
                continue
            with zip_.open(file_) as infile:
                conversation = json.load(infile)
                for id_ in conversation["responseDict"]:
                    branch = conversation["responseDict"][id_]
                    if branch["rating"] == None: # Skip unrated entries
                        continue
                    label = "Yes" if branch["rating"] else "No"
                    text = branch["evaluationPrompt"].format(
                        prompt = branch["prompt"],
                        response = branch["text"]) + "\n" + label
                    self.training_items.append(text)
        random.shuffle(self.training_items)

    def __len__(self):
        return len(self.training_items)
        
    def __next__(self):
        return random.sample(self.training_items, 1)[0]
