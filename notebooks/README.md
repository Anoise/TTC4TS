# Executed reproduction notebooks

Open these in GitHub to read the saved outputs, or install `pip install -e ".[notebooks]"` from the repository root and launch `python -m jupyterlab notebooks`.

| Notebook | Reproduces | Default compute |
|---|---|---|
| [00 CPU quickstart](00_cpu_quickstart.ipynb) | Actual fit/predict API with synthetic data | CPU, seconds |
| [01 Main results](01_main_results.ipynb) | Eleven headline aggregators, per-dataset table, seventeen baselines | Saved measured CSVs |
| [02 Generators and Adapters](02_generators_and_adapters.ipynb) | Seven generator combinations, Retrieval-demo, Risk sample sweep | Saved measured CSVs |
| [03 Candidate budgets](03_candidate_budgets.ipynb) | Six-model curves and tables for both distinct dataset versions | Saved measured CSVs |
| [04 Uncertainty](04_uncertainty.ipynb) | Thirteen methods, coverage/WIS, actual interval bands | Saved metrics/endpoints |
| [05 Forecast examples](05_forecast_examples.ipynb) | Six actual forecasts and recomputed excerpt errors | Saved numerical excerpts |
| [06 GPU reproduction](06_gpu_reproduction.ipynb) | One-cell command plus its historical reference | Offline unless RUN_GPU is changed |

Run `python tools/execute_notebooks.py` to execute all seven in order using the current Python interpreter. Execution fails on errors. All default cells are offline and require no original benchmark files or checkpoint downloads. Run all cells from a fresh kernel for deterministic snapshot analysis.

`tools/build_notebooks.py` regenerates source cells and clears outputs; run the execution command again afterwards. Editing a notebook directly is supported, but rebuilding from the source builder will replace those edits.

Tables/plots distinguish descriptive means from uncertainty estimates, unavailable oracles from deployable methods, and post-hoc forecast illustrations from aggregate evidence. GPU reproducibility prerequisites and limits are in [the guide](../docs/reproduction.md).
