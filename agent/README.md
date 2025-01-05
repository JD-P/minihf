![The ReAct loop on the left is how most LLM agents are implemented, it fails because the reasoning desynchronizes from the problem state. I attempt to fix this by having the agent write down its expectations for the action and check its work with unit test callbacks.](docs/weave_agent_vs_react.png)

## Features

Weave-Agent improves over typical [ReAct agents](https://arxiv.org/abs/2210.03629)
in the following ways:

1. Uses observation callbacks to mimic WIMP, giving it
[Cradle-like](https://baai-agents.github.io/Cradle/) ergonomics without having
to rely on a visual language model. [Smolagents](https://huggingface.co/blog/smolagents)
by contrast takes the return value of the action as its observation like a traditional
ReAct loop.
2. Writes down its expectations and uses evaluation callbacks/unit tests to check
that the things it wanted changed in the environment were actually changed by
its actions.
3. Uses Python as the trace format, so that it is always in distribution and works
like a code calling agent instead of the JSON tool calling hell people typically cope with.
4. Uses logit evaluators for search and rejection sampling so that the model can
ask itself questions and get answers and use those for e.g. flowcharts during reasoning.
5. Has a ring attention based iterated tuning loop (that's not done yet) with
careful trace design meant to ensure every cognitive ability weave-agent relies
on to function is trained by the traces.
6. Generates (up to) megabytes of coherent long text for training per session.

## Installation

This is the command I use to start vllm. `tensor-parallel-size` controls how many
GPUs you use so be sure to set it lower than 8 if you are not using a 8x H100 box.

`python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-32B-Instruct --served-model-name Qwen/Qwen2.5-Coder-32B-Instruct --max-logprobs 100 --gpu-memory-utilization=0.95 --disable-log-requests --disable-log-stats --port 5001 --tensor-parallel-size 8 --max-num-seqs 512 --enable-prefix-caching --max-model-len 131072`

If you're using `Qwen/Qwen2.5-Coder-32B-Instruct` your `config.json` in
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/<ID>/config.json`
should look like this:

```
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 27648,
  "max_position_embeddings": 32768,
  "max_window_layers": 70,
  "model_type": "qwen2",
  "num_attention_heads": 40,
  "num_hidden_layers": 64,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064,
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }

}
```

Once you have vllm running you can choose which bootstrap file you want to use
by editing the Dockerfile in this directory. To use the `vigenere_bootstrap.py`
you would set the line like so:

`CMD python weave_agent.py --port 5001 --bootstrap "bootstraps/vigenere_bootstrap.py" "Qwen/Qwen2.5-Coder-32B-Instruct" & python -m http.server 8991 --directory "/app/weave-agent-logs/"`

Then you build the docker file:

`docker buildx build -t weave-agent .`

You can run the weave-agent after building the dockerfile with this command:

`docker run -it --name weave-agent-container --network="host" weave-agent`

This starts both the agent and a web server that serves the logs directory. You
can view and save the agent-trace by visiting `http://localhost:8991/` in your
browser. An updated copy of the trace is produced on each tick.

Once you're done and want to start another run you can remove the docker container
with the following command:

`docker remove weave-agent-container`

Depending on your version of docker the command may instead be:

`docker container rm weave-agent-container`.

This should allow you to use the `docker run` command above again, or if you want
to update something you can edit the relevant files and then rebuild the docker
container.
