# Paper Workflow

Use `paper/build.sh` as the entrypoint for the paper scaffold.

## What it does

`./paper/build.sh`

That command:

1. Regenerates the architecture figure in `paper/plots/network_architecture.pdf`.
2. Regenerates the runtime plot PDF in `paper/plots/avg_attack_vs_runtime.pdf`.
3. Compiles the Typst document in `paper/typst/main.typ`.
4. Writes the PDF to `paper/paper.pdf`.

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
uv run python paper/scripts/make_avg_attack_vs_runtime_plot.py
```

Generate one runtime-vs-attack result entry from a training run:

```bash
uv run python paper/scripts/benchmark_avg_attack_vs_runtime.py \
  --run_dir training_runs/vN \
  --label "vN incumbent"
```

Only regenerate the architecture diagram:

```bash
uv run python paper/scripts/make_network_architecture_diagram.py
```

If you want Typst live-reload after the plots already exist:

```bash
typst watch --root paper paper/typst/main.typ paper/paper.pdf
```

## Layout

- `paper/build.sh`: builds plots, then compiles Typst
- `paper/plots/network_architecture.pdf`: generated architecture diagram
- `paper/results/avg_attack_vs_runtime/<entry>/summary.json`: benchmarked runtime/attack curves that feed the plot
- `paper/paper.pdf`: compiled paper output
- `paper/scripts/`: Python plot generators
- `paper/plots/`: generated figures
- `paper/typst/`: Typst sources
