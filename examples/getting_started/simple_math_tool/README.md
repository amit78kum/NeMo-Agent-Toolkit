# Simple Math Tool example

This example demonstrates creating a simple `FunctionGroup` that exposes basic
math operations (add, subtract, multiply, divide, factorial, power) for use
in NeMo Agent toolkit workflows.

Usage:

1. From the repo root, install the example or the whole repo in editable mode.

- Install this example only:

```powershell
pip install -e examples/getting_started/simple_math_tool
```

- Or install the whole repo with optional extras (recommended):

```powershell
python -m pip install -e ".[langchain]"
```

2. Run a workflow using the included config:

```powershell
nat run --config_file=examples/getting_started/simple_math_tool/src/nat_simple_math_tool/configs/config.yml --input "Add 3 and 5"
```
