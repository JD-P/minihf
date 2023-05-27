# MiniHF

MiniHF intends to be a complete human feedback pipeline for the StableLM
language models with minimal complication and dependencies. It will allow
the user to create a document context and prompt within it. Currently only
partially implemented (including the better StableLM, which is forthcoming).

## Features

- **Human Feedback**: Make your own feedback dataset by writing with local
language models such as StableLM and NeoX 20b.

- **Train A Reward Head**: Use your dataset to train a reward head on text embeddings.
The reward head allows you to automatically give your opinion on pieces of text.

- **AutoLoom**: Use your reward head to search over many possible completions
in the vein of [Janus's loom](https://generative.ink/posts/loom-interface-to-the-multiverse/).
The user does not have to manage the tree themselves, MiniHF's weave algorithm
does it for you. Simply pick a canonical entry from the top k nodes found by weave.

### Planned Features

In the future we plan to add more features to MiniHF, including:

- **Weave Hyperparameters**: Tune the parameters of the weave algorithm to tailor
your experience to your needs.

- **Tune The Base Model**: Use lora tuning to put your writing with MiniHF
into the underlying base model.

- **Bayesian Learning**: Collect feedback on the least redundant items using Bayesian active learning.

## Setup

pip install the requirements.txt and run with flask on a gpu with lots of VRAM

If your GPU does not have lots of VRAM go into the inference server source and
replace NeoX with a smaller model such as StableLM or Pythia.

## Tuning Dataset

The tuning dataset should consist of a zip file containing one or more json
conversations exported from MiniHF. You make this zip file yourself and then
upload it to the MiniHF inference server to tune the reward head.