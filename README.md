# TTC4TS: Scaling Test-Time Compute for Time Series Forecasting

PyTorch implementation, measured result snapshots, and executable notebooks accompanying the research manuscript in [paper/main.pdf](paper/main.pdf). This is a research release; it does not imply conference acceptance.

TTC4TS generates forecasts with shared-residual **Adapters**, controlled **Augmentations**, and aligned **Retrieval-demo**, then combines or selects them with nine aggregation rules. Backbone parameters remain frozen. The primary reported configuration uses **Learned weights**.

## Start here: reproduce saved results on CPU

Python 3.11 or 3.12 is recommended. From this repository directory:

```bash
python -m venv .venv
# Activate .venv with your shell's activation command.
python -m pip install -e ".[analysis,dev,notebooks]"
python scripts/reproduce_results.py
python -m jupyterlab notebooks
```

The [seven notebooks](notebooks/README.md) include executed tables and plots, so GitHub can display results without running a GPU. All run offline by default. Notebook 00 fits a small synthetic example on CPU; notebooks 01–05 recompute measured summaries from checksummed snapshots; notebook 06 also provides an explicit opt-in GPU experiment. To execute all of them:

```bash
python tools/execute_notebooks.py
python examples/synthetic_forecast.py
```

`outputs/reproduced/` receives regenerated tables and plots. These commands do not download checkpoints, overwrite the manuscript, or rerun the large GPU campaign. [Fresh GPU reproduction](docs/reproduction.md) describes those separate steps and their limitations.

## Included evidence

| Study | Completed scope | Notebook |
|---|---|---|
| Main aggregators and generator combinations | 432 cells: 6 checkpoints × 6 datasets × 4 horizons × 3 seeds | [01](notebooks/01_main_results.ipynb), [02](notebooks/02_generators_and_adapters.ipynb) |
| Independent baselines | Same 432 cells, 17 methods | [01](notebooks/01_main_results.ipynb) |
| Adapter sample count | 72 cells: 3 checkpoints × 3 datasets × 8 grids; H=96, seed 0 | [02](notebooks/02_generators_and_adapters.ipynb) |
| Fixed Learned-weights budget | 54 cells; multivariate Electricity, National Illness, Exchange Rate; 1,944 validation/test records | [03](notebooks/03_candidate_budgets.ipynb) |
| Historical single-channel budget | 54 cells; Traffic, Electricity, Exchange Rate; 3,564 test records | [03](notebooks/03_candidate_budgets.ipynb) |
| Uncertainty | 108 cells; 13 methods at 80/90/95% nominal levels; 4,212 interval records | [04](notebooks/04_uncertainty.ipynb) |
| Qualitative predictions | Six explicitly post-hoc selected forecast examples, plus measured interval excerpts | [05](notebooks/05_forecast_examples.ipynb), [04](notebooks/04_uncertainty.ipynb) |

Main datasets are ETTh1, ETTh2, ETTm1, ETTm2, Appliances, and BeijingWeather (six meteorological variables). Checkpoints are Chronos-Bolt small/base, TimesFM 2.0 500M, Timer 84M, Sundial 128M, and Chronos-2. See [models and data](docs/models-and-data.md).

The revised generator uses `max_delta=0.3`, `num_samples=20`, and Retrieval-demo top-10 only. The full pool has **80 candidates**: identity + 57 Adapter + 12 augmentation + 10 single-demo forecasts. Increasing candidates does **not** guarantee decreasing error. Negative results, unavailable test-label oracles, and calibration failures remain visible.

## Repository map

```text
ttc4ts/       PyTorch modules, forecasting API, uncertainty, snapshot readers
scripts/      Data/model downloads, GPU experiments, audits, CPU result reproduction
notebooks/    Seven executed, offline-first demonstrations and analyses
artifacts/    Measured CSV/JSON snapshots, numerical examples, SHA-256 manifest
experiments/  Machine-readable study scopes
paper/        Current LaTeX sources, tables, figures, compiled PDF
tests/        Numerical, leakage, shape, implementation, and release checks
docs/         Protocols, API, model/data provenance, release instructions
tools/        Notebook execution, integrity/privacy checks, source packaging
```


## License and citation

Project code retains the existing Apache-2.0 license. Dataset, model, and LaTeX-template terms remain separate: see [third-party notices](THIRD_PARTY_NOTICES.md). For now cite the manuscript title and this repository's release commit; author names and a permanent publication identifier should be added by the authors when available. No identities, DOI, acceptance, or public repository URL are invented here.


