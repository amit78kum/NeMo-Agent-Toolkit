# Simple String Tool example

This example demonstrates creating a simple `FunctionGroup` that exposes common
string operations (concatenate, split, count, reverse, upper, lower) for use
in NeMo Agent toolkit workflows.


Usage:

1. From the repo root, install the example or the whole repo in editable mode.

- Install this example only:

```bash
pip install -e examples/getting_started/simple_string_tool
```

- Or install the whole repo with optional extras (recommended):

```bash
python -m pip install -e ".[langchain]"
```

2. Run a workflow using the included config:

```bash
nat run --config_file=examples/getting_started/simple_string_tool/src/nat_simple_string_tool/configs/config.yml --input "Concatenate 'hello' and 'world'"
```
