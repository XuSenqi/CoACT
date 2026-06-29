# CoACT

CoACT is a observation compression framework for software engineering agents.

## Setup

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/Kndy666/CoACT.git
```

If the repository was cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

Use `.venv-vllm` for evaluation, tests, linting, local serving, and trajectory collection. Use `.venv-training` only for training workflows.
