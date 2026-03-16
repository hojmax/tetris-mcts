# Paper Workflow

Use `paper/build.sh` as the entrypoint for the paper scaffold.

## What it does

`./paper/build.sh`

That command:

1. Regenerates the dummy plot PDF in `paper/plots/bridge_vs_runtime_dummy.pdf`.
2. Compiles the starter Typst document in `paper/typst/main.typ`.
3. Writes the PDF to `paper/build/paper.pdf`.

Typst cannot run the Python plot script directly, so the shell entrypoint is the cleanest place to orchestrate both steps.

## Requirements

- `uv` for the project Python environment
- `typst` on your `PATH`

## Useful commands

Build plots and the PDF:

```bash
./paper/build.sh
```

Only regenerate the plot PDF:

```bash
uv run python paper/scripts/make_dummy_bridge_runtime_plot.py
```

If you want Typst live-reload after the plots already exist:

```bash
typst watch --root paper paper/typst/main.typ paper/build/paper.pdf
```

## Layout

- `paper/build.sh`: builds plots, then compiles Typst
- `paper/scripts/`: Python plot generators
- `paper/plots/`: generated figures
- `paper/typst/`: Typst sources
- `paper/build/`: generated PDF output
