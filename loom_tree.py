from argparse import ArgumentParser
from pathlib import Path
import json
import diff_match_patch

dmp = diff_match_patch.diff_match_patch()

class LoomTreeNode:
    def __init__(self, tree, _id, _type,
                 parent, timestamp, patch,
                 summary, cache, rating,
                 read, children):
        self.tree = tree
        self.id = _id
        self.timestamp = timestamp
        self.type = _type
        self.patch = patch
        self.summary = summary
        self.cache = cache
        self.rating = rating
        self.read = read
        try:
            self.parent = parent
        except TypeError:
            if _type == "root":
                self.parent = None
            else:
                raise ValueError
        self.children = children

    def __repr__(self):
        return self.summary + " [{}]".format(self.timestamp)
        
    def get_parent(self):
        try:
            return self.tree.node_store[self.parent]
        except (IndexError, KeyError):
            return None

    def count_children(self):
        count = 0
        queue = self.children
        while queue:
            count += 1
            child = queue.pop()
            queue += self.tree.node_store[child].children
        return count

    def count_children_decay(self, decay=0.5):
        if decay >= 1:
            raise ValueError
        count = 0
        depth = 0
        queue = [(depth, i) for i in self.children]
        while queue:
            depth, child_id = queue.pop()
            child = self.tree.node_store[child_id]
            queue += [(depth+1, i) for i in child.children]
            count += 1 * (decay ** depth)
        return count
    
    def render(self):
        if self.type == "root":
            return ""
        patches = []
        patches.append(self.patch)
        node = self
        while node.get_parent():
            node = node.get_parent()
            patches.append(node.patch)
        patches.reverse()
        out_text = ""
        for patch in patches:
            if patch == "":
                continue
            out_text = dmp.patch_apply(patch, out_text)[0]
        self.cache = out_text
        return out_text

        
class LoomTree:
    def __init__(self, trace_filepath, tokenizer=None):
        with open(trace_filepath) as infile:
            loom_tree_data = json.load(infile)
        self.filepath = trace_filepath
        self.node_store = {}
        for i in range(len(loom_tree_data["loomTree"]["nodeStore"])):
            node = loom_tree_data["loomTree"]["nodeStore"][str(i+1)]
            if node["type"] != "root":
                patch = []
                for diff_raw in node["patch"]:
                    diff = diff_match_patch.patch_obj()
                    diff.diffs = diff_raw["diffs"]
                    diff.start1 = diff_raw["start1"]
                    diff.start2 = diff_raw["start2"]
                    diff.length1 = diff_raw["length1"]
                    diff.length2 = diff_raw["length2"]
                    patch.append(diff)
            else:
                patch = ""
            self.node_store[node["id"]] = LoomTreeNode(
                self,
                node["id"],
                node["type"],
                node["parent"],
                node["timestamp"],
                patch,
                node["summary"],
                node["cache"],
                node["rating"],
                node["read"],
                node["children"],
            )
        self.tokenizer = tokenizer
        if self.tokenizer:
            for key in self.node_store.keys():
                # In theory we could speed this up by only tokenizing the diffed parts...
                text = self.node_store[key].render()
                self.node_store[key]["tokens"] = self.tokenizer(text)
        try:
            self.root = self.node_store["1"]
        except KeyError:
            self.root = self.node_store[1]

if __name__ == '__main__':
    parser = ArgumentParser()
    # TODO: Change this to a zip file of loom traces
    parser.add_argument("loom_trace", type=Path, help="The loom trace to read from")
    args = parser.parse_args()
    # Example: Retrieving training pairs for IPO/SPO
    tree = LoomTree(args.loom_trace)
    batches = []
    tree.root.score = 0
    queue = [tree.root.children,]
    while queue:
        new_batches = []
        children = queue.pop()
        children_batch = []
        for child_id in children:
            child = tree.node_store[child_id]
            child.score = child.count_children_decay(decay=0.5)
            children_batch.append(child)
            if child.type == "user" and child.get_parent().type != "root":
                parent = child.get_parent()
                grandparent = parent.get_parent()
                # Only take user edits as reward information if they share a prefix
                if (parent.render().startswith(grandparent.render())
                    and child.render().startswith(grandparent.render())):
                    child.score = child.score * 1.5
                    new_batches.append([parent, child])
            queue.append(child.children)
        if len(children_batch) > 1:
            children_batch.sort(key=lambda x: x.score)
            new_batches.append(children_batch)
        batches += new_batches
    for batch in batches:
        print(batch)
